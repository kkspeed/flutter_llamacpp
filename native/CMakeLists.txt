cmake_minimum_required(VERSION 3.14)
project("llama_bridge" C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# ---------------------------------------------------------------------------
# Locate llama.cpp submodule (one level up from native/)
# ---------------------------------------------------------------------------
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp submodule not found at ${LLAMA_CPP_DIR}. "
                        "Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------
# Platform-specific settings
# ---------------------------------------------------------------------------
if(DEFINED ANDROID_ABI)
    message(STATUS "llama_bridge: Android ABI = ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
    else()
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# Build llama.cpp as subdirectory
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL      OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" build-llama)

# ---------------------------------------------------------------------------
# Build mtmd (multimodal) from tools/mtmd/
# Match source list from llama.cpp/tools/mtmd/CMakeLists.txt
# ---------------------------------------------------------------------------
set(MTMD_DIR "${LLAMA_CPP_DIR}/tools/mtmd")

add_library(mtmd STATIC
    "${MTMD_DIR}/clip.cpp"
    "${MTMD_DIR}/mtmd.cpp"
    "${MTMD_DIR}/mtmd-audio.cpp"
    "${MTMD_DIR}/mtmd-helper.cpp"
    "${MTMD_DIR}/models/cogvlm.cpp"
    "${MTMD_DIR}/models/conformer.cpp"
    "${MTMD_DIR}/models/glm4v.cpp"
    "${MTMD_DIR}/models/internvl.cpp"
    "${MTMD_DIR}/models/kimivl.cpp"
    "${MTMD_DIR}/models/kimik25.cpp"
    "${MTMD_DIR}/models/nemotron-v2-vl.cpp"
    "${MTMD_DIR}/models/llama4.cpp"
    "${MTMD_DIR}/models/llava.cpp"
    "${MTMD_DIR}/models/minicpmv.cpp"
    "${MTMD_DIR}/models/mobilenetv5.cpp"
    "${MTMD_DIR}/models/paddleocr.cpp"
    "${MTMD_DIR}/models/pixtral.cpp"
    "${MTMD_DIR}/models/qwen2vl.cpp"
    "${MTMD_DIR}/models/qwen3vl.cpp"
    "${MTMD_DIR}/models/siglip.cpp"
    "${MTMD_DIR}/models/whisper-enc.cpp"
    "${MTMD_DIR}/models/youtuvl.cpp"
)

target_include_directories(mtmd PUBLIC
    "${MTMD_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
)
# mtmd needs access to its own parent for #include "../.." style
target_include_directories(mtmd PRIVATE
    "${LLAMA_CPP_DIR}"
    "${LLAMA_CPP_DIR}/vendor"
)

target_link_libraries(mtmd PUBLIC ggml llama)
target_compile_features(mtmd PRIVATE cxx_std_17)

if(NOT MSVC)
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

# ---------------------------------------------------------------------------
# llama_bridge library
# ---------------------------------------------------------------------------
if(APPLE)
    add_library(llama_bridge STATIC
        llama_bridge.cpp
    )
else()
    add_library(llama_bridge SHARED
        llama_bridge.cpp
    )
endif()

target_include_directories(llama_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLAMA_CPP_DIR}"
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/vendor"
    "${MTMD_DIR}"
)

target_link_libraries(llama_bridge PRIVATE
    llama
    common
    mtmd
)

# Android-specific link
if(ANDROID)
    target_link_libraries(llama_bridge PRIVATE android log)
endif()

# Ensure symbols are exported
set_target_properties(llama_bridge PROPERTIES
    C_VISIBILITY_PRESET   default
    CXX_VISIBILITY_PRESET default
)
