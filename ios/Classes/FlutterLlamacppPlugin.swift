import Flutter
import UIKit

/// Minimal plugin registration required by Flutter's plugin system.
/// All actual work goes through dart:ffi, not platform channels.
public class FlutterLlamacppPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        // No-op: this plugin uses FFI, not method channels
    }
}
