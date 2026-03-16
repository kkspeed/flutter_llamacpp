#include "llama_bridge.h"

// Anchor the bridge symbols into the final iOS binary so Dart FFI can resolve
// them through DynamicLibrary.process().
__attribute__((used))
static void * flutter_llamacpp_llama_bridge_symbols[] = {
    (void *) &llama_bridge_init,
    (void *) &llama_bridge_shutdown,
    (void *) &llama_bridge_load_model,
    (void *) &llama_bridge_free_model,
    (void *) &llama_bridge_model_info,
    (void *) &llama_bridge_load_mmproj,
    (void *) &llama_bridge_has_mmproj,
    (void *) &llama_bridge_chat_completion,
    (void *) &llama_bridge_cancel,
    (void *) &llama_bridge_count_prompt_tokens,
    (void *) &llama_bridge_clear_context,
    (void *) &llama_bridge_context_info,
    (void *) &llama_bridge_get_perf,
    (void *) &llama_bridge_free_string,
};
