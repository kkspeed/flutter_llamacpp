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
  final String requestJson;

  const _StreamWorkerArgs({
    required this.sendPort,
    required this.handleAddress,
    required this.requestJson,
  });
}

final class _StreamTokenEvent {
  final String token;

  const _StreamTokenEvent(this.token);
}

final class _StreamDoneEvent {
  final String responseJson;

  const _StreamDoneEvent(this.responseJson);
}

final class _StreamErrorEvent {
  final String error;

  const _StreamErrorEvent(this.error);
}

void _chatCompletionStreamWorker(_StreamWorkerArgs args) {
  final bindings = LlamaBindings.instance();
  final handle = Pointer<Void>.fromAddress(args.handleAddress);
  final callback = NativeCallable<TokenCallbackNative>.listener((
    Pointer<Utf8> token,
    Pointer<Void> userData,
  ) {
    args.sendPort.send(_StreamTokenEvent(token.toDartString()));
  });

  try {
    final responseJson = bindings.chatCompletion(
      handle,
      args.requestJson,
      callback.nativeFunction,
      nullptr,
    );
    args.sendPort.send(_StreamDoneEvent(responseJson));
  } catch (error) {
    args.sendPort.send(_StreamErrorEvent(error.toString()));
  } finally {
    callback.close();
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

    subscription = receivePort.listen((message) {
      if (message is _StreamTokenEvent) {
        controller.add(message.token);
        return;
      }

      if (message is _StreamDoneEvent) {
        final responseMap =
            jsonDecode(message.responseJson) as Map<String, dynamic>;
        if (responseMap.containsKey('error')) {
          controller.addError(
            Exception('Chat completion error: ${responseMap['error']}'),
          );
        } else {
          controller.add(ChatCompletionResponse.fromJson(responseMap));
        }
        subscription.cancel();
        receivePort.close();
        controller.close();
        return;
      }

      if (message is _StreamErrorEvent) {
        controller.addError(Exception(message.error));
        subscription.cancel();
        receivePort.close();
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
            requestJson: requestJson,
          ),
        );
      } catch (error) {
        await subscription.cancel();
        receivePort.close();
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
