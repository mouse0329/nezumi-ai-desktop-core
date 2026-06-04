# LiteRT-LM Integration Guide

## Overview

This document provides step-by-step instructions for completing the LiteRT-LM integration after the Bazel build completes.

## Current Status

- ✅ Bazel installed and verified (v9.1.0)
- ✅ LiteRT-LM SDK cloned to `C:\LiteRT-LM`
- ⏳ Bazel build in progress: `bazel build //runtime/engine:engine`
- ⏳ Cargo check with litert feature in progress

## What Is Being Built

The LiteRT-LM project compiles Google's edge AI inference engine optimized for running TensorFlow Lite Large Language Models on resource-constrained devices.

**Key components:**
- `runtime/engine`: Core inference engine
- `runtime/conversation`: Session/chat management
- Dependencies: TensorFlow, protobuf, gRPC, LLVM

**Build artifacts location:** `C:\LiteRT-LM\bazel-bin\runtime\`

## Post-Build Integration Steps

### 1. Verify Bazel Build Success

```powershell
# Check if engine target was built
ls C:\LiteRT-LM\bazel-bin\runtime\engine\ -R -Include *.lib,*.dll,*.a

# Expected outputs (Windows):
# - litert_lm_main_lib.lib (static library)
# - Various dependency .lib files
```

### 2. Set Environment Variable

Configure the path to the built LiteRT-LM SDK:

**Option A: System Environment Variable (Persistent)**
```powershell
[Environment]::SetEnvironmentVariable(
    "LITERT_LM_ROOT",
    "C:\LiteRT-LM",
    [EnvironmentVariableTarget]::User
)
# Restart VS Code or terminal for change to take effect
```

**Option B: Cargo Configuration (Project-Specific)**

Edit or create `.cargo/config.toml`:
```toml
[env]
LITERT_LM_ROOT = "C:/LiteRT-LM"
```

**Option C: Terminal Session (Temporary)**
```powershell
$env:LITERT_LM_ROOT = "C:\LiteRT-LM"
```

### 3. Update build.rs

The `build.rs` file needs to link against LiteRT-LM libraries. Current configuration:

```rust
fn build_litert() {
    if Path::new("native/litert_wrapper/CMakeLists.txt").exists() {
        let litert_root = env::var("LITERT_LM_ROOT")
            .unwrap_or_else(|_| "C:/LiteRT-LM".to_string());
        
        let config = cmake::Config::new("native/litert_wrapper");
        config
            .define("LITERT_LM_ROOT", &litert_root)
            .build_target("litert_wrapper")
            .build();
        
        println!("cargo:rustc-link-search=native={}\\bazel-bin\\runtime\\engine", litert_root);
        println!("cargo:rustc-link-lib=litert_lm_main_lib");
    }
}
```

### 4. Verify CMake Configuration

The CMakeLists.txt at `native/litert_wrapper/CMakeLists.txt` should:
- Find LiteRT-LM headers in `${LITERT_LM_ROOT}`
- Link against `litert_lm_main_lib` from `${LITERT_LM_ROOT}/bazel-bin/runtime/engine`

Current configuration:
```cmake
cmake_minimum_required(VERSION 3.20)
project(litert_wrapper)

set(LITERT_LM_ROOT "" CACHE PATH "Path to LiteRT-LM SDK")

add_library(litert_wrapper STATIC litert_wrapper.cpp)
target_include_directories(litert_wrapper PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${LITERT_LM_ROOT}
)
target_link_directories(litert_wrapper PRIVATE 
    ${LITERT_LM_ROOT}/bazel-bin/runtime/engine
)
target_link_libraries(litert_wrapper PRIVATE 
    litert_lm_main_lib
)
```

### 5. Test Compilation

Build the core library with litert feature enabled:

```powershell
cd C:\nezumi-ai-desktop-core

# Verify Cargo can find and link LiteRT-LM
cargo build --features litert --verbose

# If successful, test specific features
cargo test --features litert
```

Expected output:
- Compilation of `litert_wrapper` with CMake
- Linking against LiteRT-LM libraries
- No linker errors about missing symbols

### 6. Functional Testing

#### Option A: Test with TFLite Model File

```powershell
# List available models
cargo run --manifest-path cli/Cargo.toml --bin nezumiai -- list

# Import a TensorFlow Lite model
cargo run --manifest-path cli/Cargo.toml --bin nezumiai -- \
    import "C:\path\to\model.tflite" \
    --name test-litert-model

# Test generation
cargo run --manifest-path cli/Cargo.toml --bin nezumiai -- \
    run test-litert-model
```

#### Option B: Direct Rust Test

Create a test in `src/engines/litert/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_litert_engine_load() {
        let engine = LiteRTEngine::new();
        let meta = ModelMeta::from_path("/path/to/model.tflite");
        
        let result = engine.load(meta.path.as_str(), LoadConfig::default()).await;
        assert!(result.is_ok());
    }
}
```

## Troubleshooting

### Linker Error: "litert_lm_main_lib not found"

**Cause:** LITERT_LM_ROOT not set or build.rs cannot find libraries

**Solution:**
1. Verify `LITERT_LM_ROOT` environment variable is set
2. Check library exists: `ls C:\LiteRT-LM\bazel-bin\runtime\engine\*.lib`
3. Check build.rs search paths match actual location

### CMake Error: "LITERT_LM_ROOT not defined"

**Cause:** CMake cache not updated or variable not passed from build.rs

**Solution:**
1. Clean build: `cargo clean`
2. Force rebuild: `cargo build --features litert --force-rebuild`
3. Check `.cargo/config.toml` has `LITERT_LM_ROOT` defined

### Missing Header Files

**Cause:** LITERT_LM_ROOT points to wrong location or SDK not fully cloned

**Solution:**
1. Verify clone completed: `ls C:\LiteRT-LM\runtime\engine\*.h`
2. Check CMakeLists.txt include paths

### Runtime Error: "Module not found or invalid"

**Cause:** .tflite model file is invalid or incompatible

**Solution:**
1. Test with official LiteRT-LM example model first
2. Verify model is TensorFlow Lite format (`.tflite` extension)
3. Check model metadata with `tflite_interpreter_tool`

## Architecture Reference

```
┌─────────────────────────────────────────┐
│   CLI / Application Layer               │
├─────────────────────────────────────────┤
│   nezumi-ai-desktop-core (Rust)         │
├──────────────────┬──────────────────────┤
│ LlamaEngine      │ LiteRTEngine         │
│ (.gguf models)   │ (.tflite models)     │
├──────────────────┼──────────────────────┤
│ llama.cpp        │ LiteRT-LM            │
│ (C++ wrapper)    │ (Google inference)   │
├──────────────────┼──────────────────────┤
│ CUDA Runtime     │ TensorFlow Lite      │
└──────────────────┴──────────────────────┘
```

## File Reference

**Key Integration Files:**
- `src/engines/litert/mod.rs` - Rust engine implementation
- `src/ffi.rs` - C FFI declarations
- `native/litert_wrapper/litert_wrapper.cpp` - C++ wrapper (if using native implementation)
- `native/litert_wrapper/litert_wrapper.h` - C++ header
- `native/litert_wrapper/CMakeLists.txt` - CMake configuration
- `build.rs` - Build script with LiteRT-LM linking
- `.cargo/config.toml` - Environment configuration

## Next Steps After Integration

1. Create integration test with actual .tflite model
2. Benchmark performance vs llama.cpp on equivalent models
3. Add CLI commands: `show litert-model`, `benchmark`
4. Document supported model formats and quantization levels
5. Create pre-built wheel/package for distribution

## References

- [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM)
- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Google AI Edge GitHub](https://github.com/google-ai-edge)

---

**Last Updated:** 2026-06-03
**Status:** Integration in progress - awaiting Bazel build completion
