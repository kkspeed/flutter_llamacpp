#include "llama_bridge.h"

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <thread>

#if defined(__ANDROID__)
#include <android/log.h>
#include <dirent.h>
#include <dlfcn.h>
#include <libgen.h>
#define LOG_TAG "llama_bridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) fprintf(stderr, __VA_ARGS__)
#define LOGW(...) fprintf(stderr, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

// ---------------------------------------------------------------------------
// nlohmann/json (bundled with llama.cpp common/)
// ---------------------------------------------------------------------------
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Context structure
// ---------------------------------------------------------------------------
struct llama_bridge_context {
    llama_model                 * model       = nullptr;
    llama_context               * ctx         = nullptr;
    common_chat_templates_ptr     chat_tmpls;
    common_sampler              * sampler     = nullptr;
    mtmd_context                * mtmd_ctx    = nullptr;

    // Generation state
    std::vector<common_chat_msg>  chat_msgs;
    llama_pos                     n_past      = 0;
    std::atomic<bool>             cancelled{false};

    // Perf tracking
    int32_t                       perf_prompt_tokens  = 0;
    int32_t                       perf_gen_tokens     = 0;
    double                        perf_prompt_ms      = 0.0;
    double                        perf_gen_ms         = 0.0;

    // Config
    int32_t                       n_ctx       = 4096;
    int32_t                       n_batch     = 512;
    int32_t                       n_threads   = 4;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static char * strdup_c(const std::string & s) {
    char * p = (char *)malloc(s.size() + 1);
    if (p) { memcpy(p, s.c_str(), s.size() + 1); }
    return p;
}

static char * json_to_cstr(const json & j) {
    return strdup_c(j.dump());
}

#if defined(__ANDROID__)
static bool has_prefix(const std::string & s, const char * prefix) {
    return s.rfind(prefix, 0) == 0;
}

static bool has_suffix(const std::string & s, const char * suffix) {
    const size_t suffix_len = std::strlen(suffix);
    return s.size() >= suffix_len && s.compare(s.size() - suffix_len, suffix_len, suffix) == 0;
}

static bool is_dynamic_backend_library(const std::string & name) {
    if (!has_prefix(name, "libggml-") || !has_suffix(name, ".so")) {
        return false;
    }
    return name != "libggml.so" && name != "libggml-base.so";
}

static const char * const kKnownAndroidBackendSonames[] = {
    "libggml-vulkan.so",
    "libggml-cpu-android_armv9.2_2.so",
    "libggml-cpu-android_armv9.2_1.so",
    "libggml-cpu-android_armv9.0_1.so",
    "libggml-cpu-android_armv8.6_1.so",
    "libggml-cpu-android_armv8.2_2.so",
    "libggml-cpu-android_armv8.2_1.so",
    "libggml-cpu-android_armv8.0_1.so",
    "libggml-cpu.so",
};

static void try_load_android_backend(const char * name_or_path) {
    LOGI("Attempting backend load: %s\n", name_or_path);
    if (ggml_backend_load(name_or_path) != nullptr) {
        LOGI("Loaded backend: %s\n", name_or_path);
    } else {
        LOGW("Backend load failed: %s\n", name_or_path);
    }
}

static void load_android_backends_from_dir(const char * backend_dir) {
    if (!backend_dir || !backend_dir[0]) {
        return;
    }

    DIR * dir = opendir(backend_dir);
    if (!dir) {
        LOGE("Failed to open backend dir %s: %s\n", backend_dir, std::strerror(errno));
        return;
    }

    std::vector<std::string> entries;
    std::vector<std::string> candidates;
    while (dirent * entry = readdir(dir)) {
        std::string name(entry->d_name);
        entries.push_back(name);
        if (is_dynamic_backend_library(name)) {
            candidates.push_back(name);
        }
    }
    closedir(dir);

    std::sort(entries.begin(), entries.end());
    std::sort(candidates.begin(), candidates.end());

    for (const auto & entry : entries) {
        LOGI("Native lib dir entry: %s\n", entry.c_str());
    }

    if (candidates.empty()) {
        LOGW("No backend libraries visible via directory listing; trying known sonames\n");
        for (const char * soname : kKnownAndroidBackendSonames) {
            try_load_android_backend(soname);
        }
        return;
    }

    for (const auto & candidate : candidates) {
        const std::string full_path = std::string(backend_dir) + "/" + candidate;
        try_load_android_backend(full_path.c_str());
        if (ggml_backend_reg_count() == 0) {
            try_load_android_backend(candidate.c_str());
        }
    }
}
#endif

static bool is_valid_utf8(const char * s) {
    if (!s) return true;
    const unsigned char * b = (const unsigned char *)s;
    while (*b) {
        int n;
        if      ((*b & 0x80) == 0x00) n = 1;
        else if ((*b & 0xE0) == 0xC0) n = 2;
        else if ((*b & 0xF0) == 0xE0) n = 3;
        else if ((*b & 0xF8) == 0xF0) n = 4;
        else return false;
        b++;
        for (int i = 1; i < n; i++) {
            if ((*b & 0xC0) != 0x80) return false;
            b++;
        }
    }
    return true;
}

struct parsed_chat_request {
    json raw;
    std::vector<common_chat_msg> messages;
    std::vector<common_chat_tool> tools;
    bool enable_thinking = false;
};

static bool parse_chat_request(
    const char * request_json,
    parsed_chat_request & out,
    std::string & error
) {
    try {
        out.raw = json::parse(request_json);
    } catch (...) {
        error = "invalid JSON request";
        return false;
    }

    out.enable_thinking = out.raw.value("enable_thinking", false);

    if (out.raw.contains("messages") && out.raw["messages"].is_array()) {
        for (auto & m : out.raw["messages"]) {
            common_chat_msg msg;
            msg.role = m.value("role", "user");

            if (m["content"].is_string()) {
                msg.content = m["content"].get<std::string>();
            } else if (m["content"].is_array()) {
                for (auto & part : m["content"]) {
                    std::string type = part.value("type", "text");
                    if (type == "text") {
                        common_chat_msg_content_part cp;
                        cp.type = "text";
                        cp.text = part.value("text", "");
                        msg.content_parts.push_back(cp);
                    } else if (type == "image_url") {
                        common_chat_msg_content_part cp;
                        cp.type = "image_url";
                        if (part.contains("image_url")) {
                            cp.text = part["image_url"].value("url", "");
                        }
                        msg.content_parts.push_back(cp);
                    }
                }
            }

            if (m.contains("tool_call_id")) {
                msg.tool_call_id = m["tool_call_id"].get<std::string>();
            }

            out.messages.push_back(msg);
        }
    }

    if (out.raw.contains("tools") && out.raw["tools"].is_array()) {
        for (auto & t : out.raw["tools"]) {
            if (t.value("type", "") == "function" && t.contains("function")) {
                common_chat_tool tool;
                tool.name        = t["function"].value("name", "");
                tool.description = t["function"].value("description", "");
                if (t["function"].contains("parameters")) {
                    tool.parameters = t["function"]["parameters"].dump();
                }
                out.tools.push_back(tool);
            }
        }
    }

    return true;
}

static common_chat_params build_chat_params(
    llama_bridge_context * ctx,
    const parsed_chat_request & req
) {
    common_chat_templates_inputs inputs;
    inputs.messages              = req.messages;
    inputs.tools                 = req.tools;
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = req.enable_thinking;

    std::string tool_choice_str = req.raw.value("tool_choice", "auto");
    if (tool_choice_str == "none") {
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    } else if (tool_choice_str == "required") {
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    } else {
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    return common_chat_templates_apply(ctx->chat_tmpls.get(), inputs);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void llama_bridge_init(const char * backend_path) {
#if defined(__ANDROID__)
    // Android log callback
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        int prio = ANDROID_LOG_INFO;
        if (level == GGML_LOG_LEVEL_WARN)  prio = ANDROID_LOG_WARN;
        if (level == GGML_LOG_LEVEL_ERROR) prio = ANDROID_LOG_ERROR;
        __android_log_print(prio, LOG_TAG, "%s", text);
    }, nullptr);
#endif

    // Load dynamic backends if path is given.
    // On Android, infer the native library directory from libllama_bridge.so
    // so runtime CPU backend variants are available without extra Dart setup.
    if (backend_path && backend_path[0]) {
        LOGI("Loading backends from: %s\n", backend_path);
#if defined(__ANDROID__)
        load_android_backends_from_dir(backend_path);
#else
        ggml_backend_load_all_from_path(backend_path);
#endif
#if defined(__ANDROID__)
    } else {
        Dl_info info {};
        if (dladdr((void *) &llama_bridge_init, &info) != 0 && info.dli_fname != nullptr) {
            std::string so_path(info.dli_fname);
            std::vector<char> dir_buf(so_path.begin(), so_path.end());
            dir_buf.push_back('\0');
            char * so_dir = dirname(dir_buf.data());
            if (so_dir && so_dir[0]) {
                LOGI("Loading backends from inferred path: %s\n", so_dir);
                load_android_backends_from_dir(so_dir);
            }
        }
#endif
    }

    llama_backend_init();
    LOGI("llama_bridge: backend registry count = %zu\n", ggml_backend_reg_count());
    LOGI("llama_bridge: backend initialized\n");
}

void llama_bridge_shutdown(void) {
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

llama_bridge_context * llama_bridge_load_model(
    const char * model_path,
    const char * params_json
) {
    if (!model_path) return nullptr;

    auto * bctx = new llama_bridge_context();

    // Parse optional params
    int32_t n_ctx       = 4096;
    int32_t n_batch     = 512;
    int32_t n_threads   = 4;
    bool    flash_attn  = true;
    int32_t n_gpu_layers = 99; // offload as many as possible

    if (params_json && params_json[0]) {
        try {
            auto p = json::parse(params_json);
            if (p.contains("n_ctx"))        n_ctx       = p["n_ctx"].get<int32_t>();
            if (p.contains("n_batch"))      n_batch     = p["n_batch"].get<int32_t>();
            if (p.contains("n_threads"))    n_threads   = p["n_threads"].get<int32_t>();
            if (p.contains("flash_attn"))   flash_attn  = p["flash_attn"].get<bool>();
            if (p.contains("n_gpu_layers")) n_gpu_layers = p["n_gpu_layers"].get<int32_t>();
        } catch (...) {
            LOGW("llama_bridge: failed to parse params JSON, using defaults\n");
        }
    }

    if (n_threads <= 0) {
        const auto hw_threads = std::thread::hardware_concurrency();
        n_threads = hw_threads > 0 ? (int32_t) hw_threads : 4;
    }

    bctx->n_ctx    = n_ctx;
    bctx->n_batch  = n_batch;
    bctx->n_threads = n_threads;

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    LOGI("llama_bridge: loading model from %s\n", model_path);
    bctx->model = llama_model_load_from_file(model_path, model_params);
    if (!bctx->model) {
        LOGE("llama_bridge: failed to load model\n");
        delete bctx;
        return nullptr;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx         = n_ctx;
    ctx_params.n_batch       = n_batch;
    ctx_params.n_ubatch      = n_batch;
    ctx_params.n_threads     = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    bctx->ctx = llama_init_from_model(bctx->model, ctx_params);
    if (!bctx->ctx) {
        LOGE("llama_bridge: failed to create context\n");
        llama_model_free(bctx->model);
        delete bctx;
        return nullptr;
    }

    // Init chat templates
    bctx->chat_tmpls = common_chat_templates_init(bctx->model, "");

    // Init sampler with defaults
    common_params_sampling sparams;
    sparams.temp = 0.7f;
    bctx->sampler = common_sampler_init(bctx->model, sparams);

    LOGI("llama_bridge: model loaded successfully (n_ctx=%d)\n", n_ctx);
    return bctx;
}

void llama_bridge_free_model(llama_bridge_context * ctx) {
    if (!ctx) return;
    if (ctx->mtmd_ctx) { mtmd_free(ctx->mtmd_ctx); }
    if (ctx->sampler)  { common_sampler_free(ctx->sampler); }
    ctx->chat_tmpls.reset();
    if (ctx->ctx)   { llama_free(ctx->ctx); }
    if (ctx->model) { llama_model_free(ctx->model); }
    delete ctx;
}

char * llama_bridge_model_info(const llama_bridge_context * ctx) {
    if (!ctx || !ctx->model) return strdup_c("{}");

    char desc[256] = {0};
    llama_model_desc(ctx->model, desc, sizeof(desc));

    json info = {
        {"description",  desc},
        {"size_bytes",   (int64_t)llama_model_size(ctx->model)},
        {"n_params",     (int64_t)llama_model_n_params(ctx->model)},
        {"n_ctx_train",  llama_model_n_ctx_train(ctx->model)},
        {"n_ctx",        ctx->n_ctx},
        {"has_mmproj",   ctx->mtmd_ctx != nullptr},
    };
    return json_to_cstr(info);
}

// ---------------------------------------------------------------------------
// VLM / mmproj
// ---------------------------------------------------------------------------

llama_bridge_status llama_bridge_load_mmproj(
    llama_bridge_context * ctx,
    const char * mmproj_path
) {
    if (!ctx || !ctx->model || !mmproj_path) return LLAMA_BRIDGE_ERROR_NO_MODEL;

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu    = true;
    mparams.n_threads  = ctx->n_threads;

    ctx->mtmd_ctx = mtmd_init_from_file(mmproj_path, ctx->model, mparams);
    if (!ctx->mtmd_ctx) {
        LOGE("llama_bridge: failed to load mmproj from %s\n", mmproj_path);
        return LLAMA_BRIDGE_ERROR_VLM_LOAD;
    }

    LOGI("llama_bridge: mmproj loaded, vision=%d, audio=%d\n",
         mtmd_support_vision(ctx->mtmd_ctx),
         mtmd_support_audio(ctx->mtmd_ctx));
    return LLAMA_BRIDGE_OK;
}

bool llama_bridge_has_mmproj(const llama_bridge_context * ctx) {
    return ctx && ctx->mtmd_ctx != nullptr;
}

// ---------------------------------------------------------------------------
// Internal: decode tokens in batches
// ---------------------------------------------------------------------------
static int decode_batch(
    llama_context * lctx,
    const std::vector<llama_token> & tokens,
    llama_pos start_pos,
    int32_t n_batch,
    bool compute_last_logit
) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    for (int i = 0; i < (int)tokens.size(); i += n_batch) {
        int cur = std::min((int)tokens.size() - i, n_batch);
        common_batch_clear(batch);

        for (int j = 0; j < cur; j++) {
            bool logit = compute_last_logit && (i + j == (int)tokens.size() - 1);
            common_batch_add(batch, tokens[i + j], start_pos + i + j, {0}, logit);
        }

        if (llama_decode(lctx, batch) != 0) {
            llama_batch_free(batch);
            return -1;
        }
    }

    llama_batch_free(batch);
    return 0;
}

static int shift_context_window(
    llama_bridge_context * ctx,
    int32_t n_keep,
    int32_t incoming_tokens
) {
    if (ctx->n_past + incoming_tokens < ctx->n_ctx) {
        return 0;
    }

    llama_memory_t mem = llama_get_memory(ctx->ctx);
    if (!llama_memory_can_shift(mem)) {
        return -1;
    }

    while (ctx->n_past + incoming_tokens >= ctx->n_ctx) {
        const int32_t keep = std::min<int32_t>(n_keep, ctx->n_past);
        const int32_t n_left = ctx->n_past - keep;
        if (n_left <= 0) {
            return -1;
        }

        const int32_t n_discard = std::max<int32_t>(1, n_left / 2);
        LOGI("llama_bridge: shifting context (n_past=%d, keep=%d, discard=%d)\n",
             (int32_t)ctx->n_past, keep, n_discard);

        llama_memory_seq_rm (mem, 0, keep,             keep + n_discard);
        llama_memory_seq_add(mem, 0, keep + n_discard, ctx->n_past, -n_discard);

        ctx->n_past -= n_discard;
    }

    return 0;
}

static int decode_prompt_with_rolling_window(
    llama_bridge_context * ctx,
    const std::vector<llama_token> & tokens,
    int32_t n_keep
) {
    llama_batch batch = llama_batch_init(ctx->n_batch, 0, 1);

    for (int i = 0; i < (int)tokens.size(); i += ctx->n_batch) {
        int cur = std::min((int)tokens.size() - i, ctx->n_batch);
        if (shift_context_window(ctx, n_keep, cur) != 0) {
            llama_batch_free(batch);
            return -1;
        }

        common_batch_clear(batch);
        for (int j = 0; j < cur; j++) {
            bool logit = (i + j == (int)tokens.size() - 1);
            common_batch_add(batch, tokens[i + j], ctx->n_past + j, {0}, logit);
        }

        if (llama_decode(ctx->ctx, batch) != 0) {
            llama_batch_free(batch);
            return -1;
        }

        ctx->n_past += cur;
    }

    llama_batch_free(batch);
    return 0;
}

// ---------------------------------------------------------------------------
// Chat completion
// ---------------------------------------------------------------------------

char * llama_bridge_chat_completion(
    llama_bridge_context * ctx,
    const char * request_json,
    llama_bridge_token_callback callback,
    void * user_data
) {
    if (!ctx || !ctx->model || !ctx->ctx) {
        json err = {{"error", "no model loaded"}};
        return json_to_cstr(err);
    }

    // Reset cancel flag
    ctx->cancelled.store(false);

    // Parse request
    parsed_chat_request parsed_req;
    std::string parse_error;
    if (!parse_chat_request(request_json, parsed_req, parse_error)) {
        json err = {{"error", "invalid JSON request"}};
        return json_to_cstr(err);
    }
    json & req = parsed_req.raw;

    // Extract parameters
    float  temperature = req.value("temperature", 0.7f);
    float  top_p       = req.value("top_p", 0.9f);
    int    top_k       = req.value("top_k", 40);
    float  min_p       = req.value("min_p", 0.0f);
    int    max_tokens  = req.value("max_tokens", 512);
    bool   stream      = req.value("stream", true);

    // Rebuild sampler with request params
    if (ctx->sampler) { common_sampler_free(ctx->sampler); }
    common_params_sampling sparams;
    sparams.temp   = temperature;
    sparams.top_p  = top_p;
    sparams.top_k  = top_k;
    sparams.min_p  = min_p;
    ctx->sampler = common_sampler_init(ctx->model, sparams);

    common_chat_params chat_params = build_chat_params(ctx, parsed_req);

    std::string prompt = chat_params.prompt;

    // Collect image paths for VLM
    std::vector<std::string> image_paths;
    if (ctx->mtmd_ctx && req.contains("messages")) {
        for (auto & m : req["messages"]) {
            if (m["content"].is_array()) {
                for (auto & part : m["content"]) {
                    if (part.value("type", "") == "image_url" && part.contains("image_url")) {
                        std::string url = part["image_url"].value("url", "");
                        // Handle file:// URLs
                        if (url.rfind("file://", 0) == 0) {
                            url = url.substr(7);
                        }
                        if (!url.empty()) {
                            image_paths.push_back(url);
                        }
                    }
                }
            }
        }
    }

    // -- VLM path: use mtmd for image + text --
    bool use_vlm = ctx->mtmd_ctx && !image_paths.empty();

    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> prompt_tokens;
    mtmd_input_chunks * vlm_chunks = nullptr;
    int32_t n_keep = 0;

    if (use_vlm) {
        // Use mtmd tokenizer with image markers
        // Insert <__media__> markers for each image into the prompt
        std::string vlm_prompt = prompt;
        // If prompt doesn't already have markers, prepend them
        const char * marker = mtmd_default_marker();
        int marker_count = 0;
        size_t pos = 0;
        while ((pos = vlm_prompt.find(marker, pos)) != std::string::npos) {
            marker_count++;
            pos += strlen(marker);
        }
        // If we have images but no markers, inject them before the last user text
        if (marker_count < (int)image_paths.size()) {
            // Simple approach: prepend markers
            std::string prefix;
            for (int i = marker_count; i < (int)image_paths.size(); i++) {
                prefix += marker;
                prefix += "\n";
            }
            vlm_prompt = prefix + vlm_prompt;
        }

        // Load bitmaps
        std::vector<mtmd::bitmap> bitmaps;
        for (auto & path : image_paths) {
            mtmd_bitmap * raw = mtmd_helper_bitmap_init_from_file(ctx->mtmd_ctx, path.c_str());
            if (raw) {
                bitmaps.push_back(mtmd::bitmap(raw));
            } else {
                LOGW("llama_bridge: failed to load image: %s\n", path.c_str());
            }
        }

        // Prepare bitmap pointers
        std::vector<const mtmd_bitmap *> bitmap_ptrs;
        for (auto & b : bitmaps) {
            bitmap_ptrs.push_back(b.ptr.get());
        }

        // Tokenize with mtmd
        vlm_chunks = mtmd_input_chunks_init();
        mtmd_input_text input_text;
        input_text.text         = vlm_prompt.c_str();
        input_text.add_special  = true;
        input_text.parse_special = true;

        int32_t res = mtmd_tokenize(ctx->mtmd_ctx, vlm_chunks,
            &input_text, bitmap_ptrs.data(), bitmap_ptrs.size());
        if (res != 0) {
            LOGE("llama_bridge: mtmd_tokenize failed: %d\n", res);
            mtmd_input_chunks_free(vlm_chunks);
            json err = {{"error", "VLM tokenization failed"}};
            return json_to_cstr(err);
        }

        // Decode VLM chunks using helper
        llama_memory_clear(llama_get_memory(ctx->ctx), false);
        ctx->n_past = 0;

        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(ctx->mtmd_ctx, ctx->ctx, vlm_chunks,
                ctx->n_past, 0, ctx->n_batch, true, &new_n_past) != 0) {
            LOGE("llama_bridge: mtmd_helper_eval_chunks failed\n");
            mtmd_input_chunks_free(vlm_chunks);
            json err = {{"error", "VLM eval failed"}};
            return json_to_cstr(err);
        }

        ctx->n_past = new_n_past;
        if (ctx->n_past >= ctx->n_ctx) {
            mtmd_input_chunks_free(vlm_chunks);
            json err = {{"error", "prompt exceeds context window"}};
            return json_to_cstr(err);
        }

        mtmd_input_chunks_free(vlm_chunks);

    } else {
        // -- Text-only path --
        llama_memory_clear(llama_get_memory(ctx->ctx), false);
        ctx->n_past = 0;

        prompt_tokens = common_tokenize(ctx->ctx, prompt, true, true);
        n_keep = std::min<int32_t>(
            std::min<int32_t>((int32_t)prompt_tokens.size(), 256),
            std::max<int32_t>(1, ctx->n_ctx / 8));

        if (decode_prompt_with_rolling_window(ctx, prompt_tokens, n_keep) != 0) {
            json err = {{"error", "prompt decode failed"}};
            return json_to_cstr(err);
        }
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    ctx->perf_prompt_ms = std::chrono::duration<double, std::milli>(
        t_prompt_end - t_prompt_start).count();
    ctx->perf_prompt_tokens = use_vlm ? ctx->n_past : (int32_t)prompt_tokens.size();

    // -- Token generation loop --
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    std::string response_text;
    std::string utf8_buffer;
    int gen_count = 0;
    llama_batch batch = llama_batch_init(1, 0, 1);
    const llama_vocab * vocab = llama_model_get_vocab(ctx->model);

    for (int i = 0; i < max_tokens; i++) {
        if (ctx->cancelled.load()) break;
        if (!use_vlm && shift_context_window(ctx, n_keep, 1) != 0) break;
        if (use_vlm && ctx->n_past >= llama_n_ctx(ctx->ctx)) break;

        llama_token new_token = common_sampler_sample(ctx->sampler, ctx->ctx, -1);
        common_sampler_accept(ctx->sampler, new_token, true);

        // Check EOG
        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Check stop sequences
        // TODO: implement additional_stops from chat_params

        // Decode next
        common_batch_clear(batch);
        common_batch_add(batch, new_token, ctx->n_past, {0}, true);
        if (llama_decode(ctx->ctx, batch) != 0) {
            LOGE("llama_bridge: decode failed at position %d\n", ctx->n_past);
            break;
        }
        ctx->n_past++;
        gen_count++;

        // Convert to text
        std::string piece = common_token_to_piece(ctx->ctx, new_token);
        response_text += piece;

        // Stream callback with UTF-8 handling
        if (callback && stream) {
            utf8_buffer += piece;
            if (is_valid_utf8(utf8_buffer.c_str())) {
                callback(utf8_buffer.c_str(), user_data);
                utf8_buffer.clear();
            }
        }
    }

    // Flush remaining UTF-8 buffer
    if (callback && stream && !utf8_buffer.empty()) {
        callback(utf8_buffer.c_str(), user_data);
    }

    llama_batch_free(batch);

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    ctx->perf_gen_ms = std::chrono::duration<double, std::milli>(
        t_gen_end - t_gen_start).count();
    ctx->perf_gen_tokens = gen_count;

    // -- Parse tool calls from output --
    json tool_calls_json = json::array();
    std::string content_text = response_text;

    if (!parsed_req.tools.empty()) {
        common_chat_parser_params parser_params(chat_params);
        common_chat_msg parsed = common_chat_parse(response_text, false, parser_params);

        if (!parsed.tool_calls.empty()) {
            content_text = parsed.content;
            for (size_t i = 0; i < parsed.tool_calls.size(); i++) {
                auto & tc = parsed.tool_calls[i];
                tool_calls_json.push_back({
                    {"id",   tc.id.empty() ? ("call_" + std::to_string(i)) : tc.id},
                    {"type", "function"},
                    {"function", {
                        {"name",      tc.name},
                        {"arguments", tc.arguments},
                    }},
                });
            }
        }
    }

    // Build OpenAI-compatible response
    json message = {
        {"role", "assistant"},
        {"content", content_text},
    };
    if (!tool_calls_json.empty()) {
        message["tool_calls"] = tool_calls_json;
    }

    std::string finish_reason = ctx->cancelled.load() ? "stop" :
        (!tool_calls_json.empty() ? "tool_calls" :
        ((gen_count >= max_tokens || ctx->n_past >= llama_n_ctx(ctx->ctx)) ? "length" : "stop"));

    json response = {
        {"id",      "chatcmpl-local"},
        {"object",  "chat.completion"},
        {"choices", json::array({
            {
                {"index", 0},
                {"message", message},
                {"finish_reason", finish_reason},
            }
        })},
        {"usage", {
            {"prompt_tokens",     ctx->perf_prompt_tokens},
            {"completion_tokens", gen_count},
            {"total_tokens",      ctx->perf_prompt_tokens + gen_count},
        }},
    };

    return json_to_cstr(response);
}

void llama_bridge_cancel(llama_bridge_context * ctx) {
    if (ctx) ctx->cancelled.store(true);
}

int32_t llama_bridge_count_prompt_tokens(
    llama_bridge_context * ctx,
    const char * request_json
) {
    if (!ctx || !ctx->model || !ctx->ctx || !request_json) {
        return -1;
    }

    parsed_chat_request parsed_req;
    std::string parse_error;
    if (!parse_chat_request(request_json, parsed_req, parse_error)) {
        return -1;
    }

    common_chat_params chat_params = build_chat_params(ctx, parsed_req);
    std::vector<llama_token> prompt_tokens = common_tokenize(
        ctx->ctx, chat_params.prompt, true, true);
    return (int32_t)prompt_tokens.size();
}

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

void llama_bridge_clear_context(llama_bridge_context * ctx) {
    if (!ctx || !ctx->ctx) return;
    llama_memory_clear(llama_get_memory(ctx->ctx), false);
    ctx->n_past = 0;
    ctx->chat_msgs.clear();
    if (ctx->sampler) { common_sampler_reset(ctx->sampler); }
}

char * llama_bridge_context_info(const llama_bridge_context * ctx) {
    if (!ctx || !ctx->ctx) return strdup_c("{}");
    json info = {
        {"used",  ctx->n_past},
        {"total", llama_n_ctx(ctx->ctx)},
    };
    return json_to_cstr(info);
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

char * llama_bridge_get_perf(const llama_bridge_context * ctx) {
    if (!ctx) return strdup_c("{}");

    double pp_tps = ctx->perf_prompt_ms > 0
        ? (ctx->perf_prompt_tokens / (ctx->perf_prompt_ms / 1000.0)) : 0.0;
    double gen_tps = ctx->perf_gen_ms > 0
        ? (ctx->perf_gen_tokens / (ctx->perf_gen_ms / 1000.0)) : 0.0;

    json perf = {
        {"prompt_tokens_per_sec", pp_tps},
        {"gen_tokens_per_sec",    gen_tps},
        {"prompt_tokens",         ctx->perf_prompt_tokens},
        {"gen_tokens",            ctx->perf_gen_tokens},
        {"prompt_ms",             ctx->perf_prompt_ms},
        {"gen_ms",                ctx->perf_gen_ms},
    };
    return json_to_cstr(perf);
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

void llama_bridge_free_string(char * str) {
    free(str);
}
