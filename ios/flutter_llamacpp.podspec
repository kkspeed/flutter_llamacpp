Pod::Spec.new do |s|
  s.name             = 'flutter_llamacpp'
  s.version          = '0.1.0'
  s.summary          = 'Flutter plugin wrapping llama.cpp for on-device LLM inference.'
  s.description      = <<-DESC
Flutter plugin wrapping llama.cpp via dart:ffi. Provides OpenAI-compatible
chat/completion API with multi-modal and tool calling support.
                       DESC
  s.homepage         = 'https://github.com/liwen/flutter_llamacpp'
  s.license          = { :type => 'MIT' }
  s.author           = { 'liwen' => 'liwen@example.com' }
  s.source           = { :path => '.' }

  s.ios.deployment_target = '16.0'
  s.osx.deployment_target = '13.0'

  # Swift stub required by Flutter plugin system
  s.source_files = 'Classes/**/*'
  s.swift_version = '5.0'

  s.dependency 'Flutter'
  s.platform = :ios, '16.0'

  # Build llama.cpp natively via CMake at pod install time
  llama_cpp_dir = File.expand_path('../../llama.cpp', __dir__)
  native_dir    = File.expand_path('../../native', __dir__)

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    # Suppress warnings from llama.cpp
    'GCC_WARN_INHIBIT_ALL_WARNINGS' => 'YES',
    'HEADER_SEARCH_PATHS' => [
      "\"#{native_dir}\"",
      "\"#{llama_cpp_dir}/include\"",
      "\"#{llama_cpp_dir}/ggml/include\"",
      "\"#{llama_cpp_dir}/common\"",
      "\"#{llama_cpp_dir}/tools/mtmd\"",
    ].join(' '),
  }

  # Build llama.cpp + bridge via CMake script phase
  s.script_phase = {
    :name => 'Build llama_bridge via CMake',
    :script => <<-SCRIPT
      set -e
      BUILD_DIR="${PODS_TARGET_SRCROOT}/build-ios"
      NATIVE_DIR="#{native_dir}"

      CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DLLAMA_BUILD_COMMON=ON
        -DLLAMA_OPENSSL=OFF
        -DGGML_METAL=ON
        -DGGML_METAL_EMBED_LIBRARY=ON
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0
        -DCMAKE_SYSTEM_NAME=iOS
        -DCMAKE_OSX_ARCHITECTURES=arm64
      )

      cmake -B "${BUILD_DIR}" -S "${NATIVE_DIR}" "${CMAKE_ARGS[@]}"
      cmake --build "${BUILD_DIR}" --config Release -j$(sysctl -n hw.ncpu)
    SCRIPT
    ,
    :execution_position => :before_compile,
  }

  # Link the built static libraries
  s.vendored_libraries = [
    'build-ios/libllama_bridge.a',
    'build-ios/build-llama/src/libllama.a',
    'build-ios/build-llama/ggml/src/libggml.a',
    'build-ios/build-llama/ggml/src/libggml-base.a',
    'build-ios/build-llama/ggml/src/ggml-cpu/libggml-cpu.a',
    'build-ios/build-llama/ggml/src/ggml-metal/libggml-metal.a',
    'build-ios/build-llama/common/libcommon.a',
    'build-ios/libmtmd.a',
  ]

  s.frameworks = 'Metal', 'MetalKit', 'Accelerate', 'Foundation'
  s.library = 'c++'
end
