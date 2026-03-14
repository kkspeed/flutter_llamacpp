import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'ffi/llama_bindings.dart';
import 'chat_completion.dart';
import 'llama_model.dart';

final class _StreamWorkerArgs {
  final SendPort sendPort;
  final int handleAddress;
  final int callbackAddress;
  final String requestJson;

  const _StreamWorkerArgs({
    required this.sendPort,
    required this.handleAddress,
    required this.callbackAddress,
    required this.requestJson,
  });
}

void _chatCompletionStreamWorker(_StreamWorkerArgs args) {
  final bindings = LlamaBindings.instance();
  final handle = Pointer<Void>.fromAddress(args.handleAddress);
  final callback = Pointer<NativeFunction<TokenCallbackNative>>.fromAddress(
    args.callbackAddress,
  );

  try {
    final responseJson = bindings.chatCompletion(
      handle,
      args.requestJson,
      callback,
      nullptr,
    );
    args.sendPort.send({'type': 'done', 'responseJson': responseJson});
  } catch (error) {
    args.sendPort.send({'type': 'error', 'error': error.toString()});
  } finally {
    Isolate.exit();
  }
}

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
  Stream<dynamic> chatCompletionStream(ChatCompletionRequest request) {
    final controller = StreamController<dynamic>();
    final requestJson = jsonEncode(request.toJson());
    final receivePort = ReceivePort();
    late final StreamSubscription<dynamic> subscription;
    late final NativeCallable<TokenCallbackNative> callback;
    var callbackClosed = false;

    void closeCallback() {
      if (callbackClosed) return;
      callbackClosed = true;
      callback.close();
    }

    callback = NativeCallable<TokenCallbackNative>.listener((
      Pointer<Utf8> token,
      Pointer<Void> userData,
    ) {
      if (!controller.isClosed) {
        controller.add(token.toDartString());
      }
    });

    subscription = receivePort.listen((message) {
      if (message is Map && message['type'] == 'done') {
        final responseMap =
            jsonDecode(message['responseJson'] as String)
                as Map<String, dynamic>;
        if (responseMap.containsKey('error')) {
          controller.addError(
            Exception('Chat completion error: ${responseMap['error']}'),
          );
        } else {
          controller.add(ChatCompletionResponse.fromJson(responseMap));
        }
        subscription.cancel();
        receivePort.close();
        closeCallback();
        controller.close();
        return;
      }

      if (message is Map && message['type'] == 'error') {
        controller.addError(Exception(message['error'] as String));
        subscription.cancel();
        receivePort.close();
        closeCallback();
        controller.close();
      }
    });

    () async {
      try {
        await Isolate.spawn(
          _chatCompletionStreamWorker,
          _StreamWorkerArgs(
            sendPort: receivePort.sendPort,
            handleAddress: _model.handle.address,
            callbackAddress: callback.nativeFunction.address,
            requestJson: requestJson,
          ),
        );
      } catch (error) {
        await subscription.cancel();
        receivePort.close();
        closeCallback();
        controller.addError(error);
        await controller.close();
      }
    }();

    return controller.stream;
  }

  /// Cancel an ongoing generation. Thread-safe.
  void cancel() {
    _bindings.cancel(_model.handle);
  }

  /// Count prompt tokens for a chat request after applying the model's
  /// chat template.
  int countPromptTokens(ChatCompletionRequest request) {
    final requestJson = jsonEncode(request.toJson());
    final count = _bindings.countPromptTokens(_model.handle, requestJson);
    if (count < 0) {
      throw Exception('Failed to count prompt tokens');
    }
    return count;
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
