# flutter_llamacpp

A Flutter plugin wrapping [llama.cpp](https://github.com/ggerganov/llama.cpp) for **on-device LLM inference** with an OpenAI-compatible chat/completion API.

## Features

- **OpenAI-compatible API** — `ChatCompletionRequest`/`Response` matching the `/chat/completions` format
- **Multi-modal (VLM)** — Image + text input via vision projector (mmproj)
- **Tool calling** — Function calling with automatic parsing via llama.cpp's built-in PEG parser
- **Streaming** — Token-by-token generation with cancel support
- **Platform optimized**:
  - **Android**: CPU by default, optional Vulkan GPU backend
  - **iOS**: CPU + Metal GPU (embedded shaders)
- **Dart FFI** — Direct native binding, no platform channels overhead
- **Tested with**: Qwen 3.5 0.8B Q4 (native multi-modal with gated attention)

## Quick Start

```dart
import 'package:flutter_llamacpp/flutter_llamacpp.dart';

// Initialize backend (once at app startup)
LlamaModel.initBackend();

// Load model
final model = LlamaModel.load('/path/to/Qwen3.5-0.8B-Q4_K_S.gguf');

// Load vision projector for multi-modal (optional)
model.loadVisionProjector('/path/to/mmproj-F16.gguf');

// Create engine
final engine = LlamaEngine(model);

// Run chat completion
final response = await engine.chatCompletion(
  ChatCompletionRequest(
    messages: [
      ChatMessage.system('You are a helpful assistant.'),
      ChatMessage.user('What is Flutter?'),
    ],
  ),
);

print(response.content);
print(engine.lastPerf); // tokens/sec stats

// Multi-modal example
final vlmResponse = await engine.chatCompletion(
  ChatCompletionRequest(
    messages: [
      ChatMessage.userMultiModal([
        ContentPart.text('What is in this image?'),
        ContentPart.imageFile('/path/to/photo.jpg'),
      ]),
    ],
  ),
);

// Tool calling example
final toolResponse = await engine.chatCompletion(
  ChatCompletionRequest(
    messages: [ChatMessage.user('What is the weather in NYC?')],
    tools: [
      Tool(function: FunctionDef(
        name: 'get_weather',
        description: 'Get current weather',
        parameters: {
          'type': 'object',
          'properties': {
            'location': {'type': 'string', 'description': 'City name'},
          },
          'required': ['location'],
        },
      )),
    ],
  ),
);

// Cleanup
model.dispose();
LlamaModel.shutdownBackend();
```

## Project Structure

```
flutter_llamacpp/
├── lib/                          # Dart API
│   ├── flutter_llamacpp.dart     # Barrel export
│   └── src/
│       ├── llama_model.dart      # Model loading & management
│       ├── llama_engine.dart     # Chat completion engine
│       ├── chat_message.dart     # Message & tool types
│       ├── chat_completion.dart  # Request/response types
│       └── ffi/
│           ├── llama_bindings.dart  # FFI bindings
│           └── llama_library.dart   # Platform library loader
├── native/                       # C++ bridge
│   ├── llama_bridge.h            # C ABI header
│   ├── llama_bridge.cpp          # Implementation
│   └── CMakeLists.txt            # Build config
├── android/                      # Android plugin
├── ios/                          # iOS plugin (CocoaPods)
├── llama.cpp/                    # Submodule
└── example/                      # Example app
```

## Requirements

- Flutter ≥ 3.29.0
- Android NDK (for Android builds)
- Xcode (for iOS builds)
- CMake ≥ 3.14

## Android Vulkan

Android builds now try to enable Vulkan automatically when a usable `glslc`
is found. The Android NDK already ships `glslc`, and the build checks the
active NDK under `shader-tools/<host>/glslc` first. You only need to set a
custom `glslc` path if you want to override that default.

You can override detection in `android/gradle.properties` or
`example/android/gradle.properties`:

```bash
flutterLlamacppEnableVulkan=true
flutterLlamacppGlslc=/absolute/path/to/glslc
```

or environment variables:

```bash
export FLUTTER_LLAMACPP_ENABLE_VULKAN=true
export VULKAN_GLSLC_EXECUTABLE=/absolute/path/to/glslc
flutter run --release
```

If `VULKAN_SDK` is set, the Android build will also look for `glslc` under
`$VULKAN_SDK/bin/glslc`.

Detection order for `glslc` is:
1. `flutterLlamacppGlslc`
2. `VULKAN_GLSLC_EXECUTABLE`
3. Android NDK `shader-tools/<host>/glslc`
4. `VULKAN_SDK/bin/glslc`

Build behavior is:
1. If `flutterLlamacppEnableVulkan` or `FLUTTER_LLAMACPP_ENABLE_VULKAN` is set, that value wins.
2. Otherwise Vulkan is enabled automatically when `glslc` is found.
3. If no usable `glslc` path is found, the plugin falls back to the CPU backend.

## Models

Download GGUF models from HuggingFace. For Qwen 3.5 0.8B:
- **Text model**: [Qwen3.5-0.8B-Q4_K_S.gguf](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_S.gguf)
- **Vision projector**: [mmproj-F16.gguf](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf)

## License

MIT
