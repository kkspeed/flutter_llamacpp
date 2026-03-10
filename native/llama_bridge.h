#pragma once

/**
 * llama_bridge - Flat C API for Flutter FFI binding to llama.cpp
 *
 * Provides OpenAI-compatible chat/completion with multi-modal (VLM)
 * and tool calling support. All strings are UTF-8.
 * Returned strings must be freed with llama_bridge_free_string().
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------
typedef enum {
    LLAMA_BRIDGE_OK                  = 0,
    LLAMA_BRIDGE_ERROR_LOAD_FAILED   = 1,
    LLAMA_BRIDGE_ERROR_CONTEXT_FAIL  = 2,
    LLAMA_BRIDGE_ERROR_NO_MODEL      = 3,
    LLAMA_BRIDGE_ERROR_DECODE        = 4,
    LLAMA_BRIDGE_ERROR_CANCELLED     = 5,
    LLAMA_BRIDGE_ERROR_INVALID_JSON  = 6,
    LLAMA_BRIDGE_ERROR_VLM_LOAD      = 7,
    LLAMA_BRIDGE_ERROR_VLM_ENCODE    = 8,
    LLAMA_BRIDGE_ERROR_TOKENIZE      = 9,
} llama_bridge_status;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
typedef struct llama_bridge_context llama_bridge_context;

// ---------------------------------------------------------------------------
// Token streaming callback
// Called for each generated token. Return false to stop generation.
// ---------------------------------------------------------------------------
typedef bool (*llama_bridge_token_callback)(const char * token_text, void * user_data);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Initialize the llama.cpp backend. Call once at app startup.
/// `backend_path` is the directory containing backend shared libs (Android NDK
/// path for dynamic backend loading). Pass NULL on iOS or if not using
/// dynamic backend loading.
void llama_bridge_init(const char * backend_path);

/// Shut down the backend. Call once at app exit.
void llama_bridge_shutdown(void);

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

/// Load a GGUF model from file path.
/// `params_json` is an optional JSON string with model parameters:
///   { "n_ctx": 4096, "n_batch": 512, "n_threads": 4, "flash_attn": true }
/// Pass NULL for defaults.
/// Returns a context handle, or NULL on failure.
llama_bridge_context * llama_bridge_load_model(
    const char * model_path,
    const char * params_json
);

/// Free model and all associated resources.
void llama_bridge_free_model(llama_bridge_context * ctx);

/// Get model metadata as JSON string. Caller must free with llama_bridge_free_string().
/// Returns: { "name": "...", "size": 123, "n_params": 123, "n_ctx_train": 4096, ... }
char * llama_bridge_model_info(const llama_bridge_context * ctx);

// ---------------------------------------------------------------------------
// Multi-modal (VLM) - Vision projector
// ---------------------------------------------------------------------------

/// Load a vision projector (mmproj GGUF) for multi-modal support.
/// Must be called after llama_bridge_load_model().
/// Returns status code.
llama_bridge_status llama_bridge_load_mmproj(
    llama_bridge_context * ctx,
    const char * mmproj_path
);

/// Check if VLM is loaded and ready.
bool llama_bridge_has_mmproj(const llama_bridge_context * ctx);

// ---------------------------------------------------------------------------
// Chat completion (OpenAI-compatible)
// ---------------------------------------------------------------------------

/// Run a chat completion request.
///
/// `request_json` is an OpenAI-format JSON string:
/// {
///   "messages": [
///     {"role": "system", "content": "You are helpful."},
///     {"role": "user", "content": "Hello"},
///     {"role": "user", "content": [
///       {"type": "text", "text": "Describe this image"},
///       {"type": "image_url", "image_url": {"url": "file:///path/to/img.jpg"}}
///     ]}
///   ],
///   "temperature": 0.7,
///   "top_p": 0.9,
///   "top_k": 40,
///   "max_tokens": 512,
///   "tools": [{"type": "function", "function": {"name": "...", ...}}],
///   "tool_choice": "auto",
///   "stream": true
/// }
///
/// If `callback` is non-NULL and stream is true, it is called for each token.
/// Returns the complete response as an OpenAI-format JSON string.
/// Caller must free with llama_bridge_free_string().
///
/// Response format:
/// {
///   "id": "chatcmpl-...",
///   "object": "chat.completion",
///   "choices": [{
///     "index": 0,
///     "message": {
///       "role": "assistant",
///       "content": "...",
///       "tool_calls": [{"id": "...", "type": "function", "function": {...}}]
///     },
///     "finish_reason": "stop"
///   }],
///   "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
/// }
char * llama_bridge_chat_completion(
    llama_bridge_context * ctx,
    const char * request_json,
    llama_bridge_token_callback callback,
    void * user_data
);

/// Cancel an ongoing generation. Thread-safe.
void llama_bridge_cancel(llama_bridge_context * ctx);

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

/// Clear the conversation context (KV cache).
void llama_bridge_clear_context(llama_bridge_context * ctx);

/// Get context usage info as JSON.
/// Returns: { "used": 123, "total": 4096 }
char * llama_bridge_context_info(const llama_bridge_context * ctx);

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

/// Get performance stats from the last generation as JSON.
/// Returns: { "prompt_tokens_per_sec": 123.4, "gen_tokens_per_sec": 45.6,
///            "prompt_tokens": 10, "gen_tokens": 20,
///            "prompt_ms": 100.0, "gen_ms": 500.0 }
char * llama_bridge_get_perf(const llama_bridge_context * ctx);

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Free a string returned by any llama_bridge_*() function.
void llama_bridge_free_string(char * str);

#ifdef __cplusplus
}
#endif
