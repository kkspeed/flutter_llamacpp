import 'dart:convert';
import 'dart:ffi';

import 'ffi/llama_bindings.dart';

/// Configuration parameters for loading a model.
class ModelParams {
  /// Context window size (0 = model default).
  final int nCtx;

  /// Batch size for prompt processing.
  final int nBatch;

  /// Number of threads for inference (0 = auto).
  final int nThreads;

  /// Use flash attention if available.
  final bool flashAttn;

  /// Number of layers to offload to GPU (-1 = all).
  final int nGpuLayers;

  const ModelParams({
    this.nCtx = 4096,
    this.nBatch = 512,
    this.nThreads = 4,
    this.flashAttn = true,
    this.nGpuLayers = 99,
  });

  String toJson() => jsonEncode({
        'n_ctx': nCtx,
        'n_batch': nBatch,
        'n_threads': nThreads,
        'flash_attn': flashAttn,
        'n_gpu_layers': nGpuLayers,
      });
}

/// Represents a loaded LLM model.
///
/// Wraps the native llama_bridge_context handle and provides
/// model lifecycle management.
class LlamaModel {
  final Pointer<Void> _handle;
  final LlamaBindings _bindings;
  bool _disposed = false;

  LlamaModel._(this._handle, this._bindings);

  /// Initialize the llama.cpp backend. Call once at app startup.
  ///
  /// [backendPath] is the directory containing backend shared libraries.
  /// On Android, this is typically the native library directory.
  /// Pass null on iOS or if not using dynamic backend loading.
  static void initBackend({String? backendPath}) {
    LlamaBindings.instance().init(backendPath: backendPath);
  }

  /// Shut down the backend. Call once at app exit.
  static void shutdownBackend() {
    LlamaBindings.instance().shutdown();
  }

  /// Load a GGUF model from a file path.
  ///
  /// Returns a [LlamaModel] instance on success.
  /// Throws [Exception] if the model fails to load.
  static LlamaModel load(String path, {ModelParams params = const ModelParams()}) {
    final bindings = LlamaBindings.instance();
    final handle = bindings.loadModel(path, paramsJson: params.toJson());

    if (handle == nullptr) {
      throw Exception('Failed to load model from: $path');
    }

    return LlamaModel._(handle, bindings);
  }

  /// Load a vision projector (mmproj GGUF) for multi-modal support.
  ///
  /// Must be called after the model is loaded.
  /// Throws [Exception] on failure.
  void loadVisionProjector(String mmprojPath) {
    _ensureNotDisposed();
    final status = _bindings.loadMmproj(_handle, mmprojPath);
    if (status != 0) {
      throw Exception('Failed to load vision projector from: $mmprojPath (status: $status)');
    }
  }

  /// Whether a vision projector is loaded (multi-modal ready).
  bool get hasVisionProjector {
    _ensureNotDisposed();
    return _bindings.hasMmproj(_handle);
  }

  /// Get model metadata.
  Map<String, dynamic> get info {
    _ensureNotDisposed();
    final jsonStr = _bindings.modelInfo(_handle);
    return jsonDecode(jsonStr) as Map<String, dynamic>;
  }

  /// The native handle for use with [LlamaEngine].
  Pointer<Void> get handle {
    _ensureNotDisposed();
    return _handle;
  }

  /// Release all native resources.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _bindings.freeModel(_handle);
  }

  void _ensureNotDisposed() {
    if (_disposed) {
      throw StateError('LlamaModel has been disposed');
    }
  }
}
