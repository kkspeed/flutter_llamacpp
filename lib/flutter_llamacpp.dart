/// Flutter plugin wrapping llama.cpp for on-device LLM inference.
///
/// Provides an OpenAI-compatible chat/completion API with multi-modal
/// (vision) and tool calling support. Targets Android (CPU + OpenCL/Vulkan GPU)
/// and iOS (Metal GPU by default with automatic CPU fallback).
///
/// ## Quick Start
///
/// ```dart
/// import 'package:flutter_llamacpp/flutter_llamacpp.dart';
///
/// // Initialize backend (once at app startup)
/// LlamaModel.initBackend();
///
/// // Load model
/// final model = LlamaModel.load('/path/to/model.gguf');
///
/// // Optionally load vision projector for multi-modal
/// model.loadVisionProjector('/path/to/mmproj.gguf');
///
/// // Create engine and run completion
/// final engine = LlamaEngine(model);
/// final response = await engine.chatCompletion(
///   ChatCompletionRequest(
///     messages: [
///       ChatMessage.system('You are a helpful assistant.'),
///       ChatMessage.user('Hello!'),
///     ],
///   ),
/// );
///
/// print(response.content);
///
/// // Cleanup
/// model.dispose();
/// LlamaModel.shutdownBackend();
/// ```

library;

export 'src/llama_model.dart';
export 'src/llama_engine.dart';
export 'src/chat_message.dart';
export 'src/chat_completion.dart';
