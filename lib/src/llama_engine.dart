import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';



import 'ffi/llama_bindings.dart';
import 'chat_completion.dart';

import 'llama_model.dart';

/// Main engine for running chat completions with a loaded LLM.
///
/// Provides both synchronous (full response) and streaming
/// (token-by-token) chat completion APIs.
class LlamaEngine {
  final LlamaModel _model;
  final LlamaBindings _bindings;

  LlamaEngine(this._model) : _bindings = LlamaBindings.instance();

  /// Run a chat completion and return the full response.
  ///
  /// This runs the generation on a background isolate to avoid
  /// blocking the UI thread.
  Future<ChatCompletionResponse> chatCompletion(
    ChatCompletionRequest request,
  ) async {
    final requestJson = jsonEncode(request.toJson());
    final handle = _model.handle;

    // Run generation on a background isolate
    final responseJson = await Isolate.run(() {
      final bindings = LlamaBindings.instance();
      return bindings.chatCompletion(
        handle,
        requestJson,
        nullptr, // no streaming callback
        nullptr,
      );
    });

    final responseMap = jsonDecode(responseJson) as Map<String, dynamic>;

    if (responseMap.containsKey('error')) {
      throw Exception('Chat completion error: ${responseMap['error']}');
    }

    return ChatCompletionResponse.fromJson(responseMap);
  }

  /// Run a streaming chat completion, yielding tokens as they are generated.
  ///
  /// The stream emits individual token strings. The final
  /// [ChatCompletionResponse] can be obtained by awaiting the
  /// [StreamSubscription] to complete.
  ///
  /// Example:
  /// ```dart
  /// final stream = engine.chatCompletionStream(request);
  /// ChatCompletionResponse? response;
  /// await for (final event in stream) {
  ///   if (event is String) {
  ///     // Token text
  ///     print(event);
  ///   } else if (event is ChatCompletionResponse) {
  ///     // Final response
  ///     response = event;
  ///   }
  /// }
  /// ```
  Stream<dynamic> chatCompletionStream(
    ChatCompletionRequest request,
  ) {
    final controller = StreamController<dynamic>();
    final requestJson = jsonEncode(request.toJson());
    final handle = _model.handle;

    // Use a ReceivePort for token callbacks from the native side
    final receivePort = ReceivePort();

    // Run generation in a background isolate
    Isolate.run(() {
      final bindings = LlamaBindings.instance();

      // For streaming, we use the non-callback path and let
      // the response be returned. The native callback mechanism
      // requires NativeCallable which must be on the same thread.
      // Instead, we use the non-streaming path and parse the result.
      // TODO: Implement proper NativeCallable streaming for real-time tokens
      return bindings.chatCompletion(
        handle,
        requestJson,
        nullptr,
        nullptr,
      );
    }).then((responseJson) {
      final responseMap = jsonDecode(responseJson) as Map<String, dynamic>;

      if (responseMap.containsKey('error')) {
        controller.addError(Exception('Chat completion error: ${responseMap['error']}'));
      } else {
        final response = ChatCompletionResponse.fromJson(responseMap);

        // Emit the full content as if it were streamed
        // (real streaming will be implemented with NativeCallable)
        final content = response.content;
        if (content != null && content.isNotEmpty) {
          controller.add(content);
        }
        controller.add(response);
      }

      receivePort.close();
      controller.close();
    }).catchError((Object error) {
      controller.addError(error);
      receivePort.close();
      controller.close();
    });

    return controller.stream;
  }

  /// Cancel an ongoing generation. Thread-safe.
  void cancel() {
    _bindings.cancel(_model.handle);
  }

  /// Clear the conversation context (KV cache).
  void clearContext() {
    _bindings.clearContext(_model.handle);
  }

  /// Get context usage information.
  Map<String, dynamic> get contextInfo {
    final jsonStr = _bindings.contextInfo(_model.handle);
    return jsonDecode(jsonStr) as Map<String, dynamic>;
  }

  /// Get performance statistics from the last generation.
  PerformanceStats get lastPerf {
    final jsonStr = _bindings.getPerf(_model.handle);
    return PerformanceStats.fromJson(
      jsonDecode(jsonStr) as Map<String, dynamic>,
    );
  }
}
