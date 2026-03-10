import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'llama_library.dart';

// ---------------------------------------------------------------------------
// Native function typedefs
// ---------------------------------------------------------------------------

// void llama_bridge_init(const char * backend_path)
typedef _InitNative = Void Function(Pointer<Utf8> backendPath);
typedef _InitDart = void Function(Pointer<Utf8> backendPath);

// void llama_bridge_shutdown()
typedef _ShutdownNative = Void Function();
typedef _ShutdownDart = void Function();

// llama_bridge_context * llama_bridge_load_model(const char * path, const char * params_json)
typedef _LoadModelNative = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> paramsJson);
typedef _LoadModelDart = Pointer<Void> Function(
    Pointer<Utf8> modelPath, Pointer<Utf8> paramsJson);

// void llama_bridge_free_model(llama_bridge_context * ctx)
typedef _FreeModelNative = Void Function(Pointer<Void> ctx);
typedef _FreeModelDart = void Function(Pointer<Void> ctx);

// char * llama_bridge_model_info(const llama_bridge_context * ctx)
typedef _ModelInfoNative = Pointer<Utf8> Function(Pointer<Void> ctx);
typedef _ModelInfoDart = Pointer<Utf8> Function(Pointer<Void> ctx);

// int llama_bridge_load_mmproj(llama_bridge_context * ctx, const char * path)
typedef _LoadMmprojNative = Int32 Function(
    Pointer<Void> ctx, Pointer<Utf8> path);
typedef _LoadMmprojDart = int Function(Pointer<Void> ctx, Pointer<Utf8> path);

// bool llama_bridge_has_mmproj(const llama_bridge_context * ctx)
typedef _HasMmprojNative = Bool Function(Pointer<Void> ctx);
typedef _HasMmprojDart = bool Function(Pointer<Void> ctx);

// Token callback: bool (*)(const char * token, void * user_data)
typedef TokenCallbackNative = Bool Function(
    Pointer<Utf8> token, Pointer<Void> userData);

// char * llama_bridge_chat_completion(ctx, request_json, callback, user_data)
typedef _ChatCompletionNative = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Utf8> requestJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData);
typedef _ChatCompletionDart = Pointer<Utf8> Function(
    Pointer<Void> ctx,
    Pointer<Utf8> requestJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData);

// void llama_bridge_cancel(llama_bridge_context * ctx)
typedef _CancelNative = Void Function(Pointer<Void> ctx);
typedef _CancelDart = void Function(Pointer<Void> ctx);

// void llama_bridge_clear_context(llama_bridge_context * ctx)
typedef _ClearContextNative = Void Function(Pointer<Void> ctx);
typedef _ClearContextDart = void Function(Pointer<Void> ctx);

// char * llama_bridge_context_info(const llama_bridge_context * ctx)
typedef _ContextInfoNative = Pointer<Utf8> Function(Pointer<Void> ctx);
typedef _ContextInfoDart = Pointer<Utf8> Function(Pointer<Void> ctx);

// char * llama_bridge_get_perf(const llama_bridge_context * ctx)
typedef _GetPerfNative = Pointer<Utf8> Function(Pointer<Void> ctx);
typedef _GetPerfDart = Pointer<Utf8> Function(Pointer<Void> ctx);

// void llama_bridge_free_string(char * str)
typedef _FreeStringNative = Void Function(Pointer<Utf8> str);
typedef _FreeStringDart = void Function(Pointer<Utf8> str);

// ---------------------------------------------------------------------------
// Bindings class
// ---------------------------------------------------------------------------

/// Low-level FFI bindings to the llama_bridge native library.
class LlamaBindings {
  static LlamaBindings? _instance;

  final DynamicLibrary _lib;

  late final _InitDart _init;
  late final _ShutdownDart _shutdown;
  late final _LoadModelDart _loadModel;
  late final _FreeModelDart _freeModel;
  late final _ModelInfoDart _modelInfo;
  late final _LoadMmprojDart _loadMmproj;
  late final _HasMmprojDart _hasMmproj;
  late final _ChatCompletionDart _chatCompletion;
  late final _CancelDart _cancel;
  late final _ClearContextDart _clearContext;
  late final _ContextInfoDart _contextInfo;
  late final _GetPerfDart _getPerf;
  late final _FreeStringDart _freeString;

  LlamaBindings._(this._lib) {
    _init = _lib
        .lookupFunction<_InitNative, _InitDart>('llama_bridge_init');
    _shutdown = _lib
        .lookupFunction<_ShutdownNative, _ShutdownDart>('llama_bridge_shutdown');
    _loadModel = _lib
        .lookupFunction<_LoadModelNative, _LoadModelDart>('llama_bridge_load_model');
    _freeModel = _lib
        .lookupFunction<_FreeModelNative, _FreeModelDart>('llama_bridge_free_model');
    _modelInfo = _lib
        .lookupFunction<_ModelInfoNative, _ModelInfoDart>('llama_bridge_model_info');
    _loadMmproj = _lib
        .lookupFunction<_LoadMmprojNative, _LoadMmprojDart>('llama_bridge_load_mmproj');
    _hasMmproj = _lib
        .lookupFunction<_HasMmprojNative, _HasMmprojDart>('llama_bridge_has_mmproj');
    _chatCompletion = _lib
        .lookupFunction<_ChatCompletionNative, _ChatCompletionDart>('llama_bridge_chat_completion');
    _cancel = _lib
        .lookupFunction<_CancelNative, _CancelDart>('llama_bridge_cancel');
    _clearContext = _lib
        .lookupFunction<_ClearContextNative, _ClearContextDart>('llama_bridge_clear_context');
    _contextInfo = _lib
        .lookupFunction<_ContextInfoNative, _ContextInfoDart>('llama_bridge_context_info');
    _getPerf = _lib
        .lookupFunction<_GetPerfNative, _GetPerfDart>('llama_bridge_get_perf');
    _freeString = _lib
        .lookupFunction<_FreeStringNative, _FreeStringDart>('llama_bridge_free_string');
  }

  /// Get the singleton instance, loading the library if needed.
  factory LlamaBindings.instance() {
    _instance ??= LlamaBindings._(loadLlamaBridgeLibrary());
    return _instance!;
  }

  // -- Public API --

  void init({String? backendPath}) {
    final bp = backendPath?.toNativeUtf8() ?? nullptr.cast<Utf8>();
    _init(bp);
    if (backendPath != null) calloc.free(bp);
  }

  void shutdown() => _shutdown();

  Pointer<Void> loadModel(String modelPath, {String? paramsJson}) {
    final mp = modelPath.toNativeUtf8();
    final pj = paramsJson?.toNativeUtf8() ?? nullptr.cast<Utf8>();
    final result = _loadModel(mp, pj);
    calloc.free(mp);
    if (paramsJson != null) calloc.free(pj);
    return result;
  }

  void freeModel(Pointer<Void> ctx) => _freeModel(ctx);

  String modelInfo(Pointer<Void> ctx) {
    final ptr = _modelInfo(ctx);
    final result = ptr.toDartString();
    _freeString(ptr);
    return result;
  }

  int loadMmproj(Pointer<Void> ctx, String path) {
    final pp = path.toNativeUtf8();
    final result = _loadMmproj(ctx, pp);
    calloc.free(pp);
    return result;
  }

  bool hasMmproj(Pointer<Void> ctx) => _hasMmproj(ctx);

  String chatCompletion(
    Pointer<Void> ctx,
    String requestJson,
    Pointer<NativeFunction<TokenCallbackNative>> callback,
    Pointer<Void> userData,
  ) {
    final rj = requestJson.toNativeUtf8();
    final ptr = _chatCompletion(ctx, rj, callback, userData);
    calloc.free(rj);
    final result = ptr.toDartString();
    _freeString(ptr);
    return result;
  }

  void cancel(Pointer<Void> ctx) => _cancel(ctx);
  void clearContext(Pointer<Void> ctx) => _clearContext(ctx);

  String contextInfo(Pointer<Void> ctx) {
    final ptr = _contextInfo(ctx);
    final result = ptr.toDartString();
    _freeString(ptr);
    return result;
  }

  String getPerf(Pointer<Void> ctx) {
    final ptr = _getPerf(ctx);
    final result = ptr.toDartString();
    _freeString(ptr);
    return result;
  }
}
