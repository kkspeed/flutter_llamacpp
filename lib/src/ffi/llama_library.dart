import 'dart:ffi';
import 'dart:io';

/// Platform-aware loader for the llama_bridge native library.
DynamicLibrary loadLlamaBridgeLibrary() {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libllama_bridge.so');
  } else if (Platform.isIOS) {
    // On iOS, static libraries are linked into the process.
    return DynamicLibrary.process();
  } else if (Platform.isMacOS) {
    return DynamicLibrary.process();
  } else {
    throw UnsupportedError(
      'flutter_llamacpp is not supported on ${Platform.operatingSystem}',
    );
  }
}
