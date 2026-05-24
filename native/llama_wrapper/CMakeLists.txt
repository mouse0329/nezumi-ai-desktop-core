cmake_minimum_required(VERSION 3.14)
project(nezumi_llama_wrapper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- GPU オプション（cargo feature から cmake -D で渡す） ---
option(LLAMA_CUDA   "Enable CUDA"   OFF)
option(LLAMA_METAL  "Enable Metal"  OFF)
option(LLAMA_VULKAN "Enable Vulkan" OFF)

if(LLAMA_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(LLAMA_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(LLAMA_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

# llama.cpp 本体
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
add_subdirectory(llama.cpp)

# ラッパー静的ライブラリ
add_library(nezumi_llama_wrapper STATIC
    llama_wrapper.cpp
    llama_wrapper.h
)

target_include_directories(nezumi_llama_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
)

# cmake-rs は MSVC で /EHsc を渡さないため、ターゲットスコープで明示付与
target_compile_options(nezumi_llama_wrapper PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
)

target_link_libraries(nezumi_llama_wrapper PRIVATE llama)
