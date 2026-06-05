use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    build_llama(&target);
    if env::var("CARGO_FEATURE_LITERT").is_ok() {
        configure_litert_runtime();
    }
}

fn litert_lm_root() -> PathBuf {
    if let Ok(root) = env::var("LITERT_LM_ROOT") {
        return PathBuf::from(root);
    }
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest).join("LiteRT-LM")
}

fn detect_bazel_execroot(litert_root: &Path) -> Option<PathBuf> {
    if let Ok(execroot) = env::var("LITERT_LM_BAZEL_EXECROOT") {
        let p = PathBuf::from(execroot);
        if p.exists() {
            return Some(p);
        }
    }
    for name in ["bazel-LiteRT-LM", "bazel-litert-lm", "bazel-LiteRT-lm"] {
        let candidate = litert_root.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn build_llama(target: &str) {
    let src = Path::new("native/llama_wrapper");

    let mut cfg = cmake::Config::new(src);
    cfg.define("CMAKE_BUILD_TYPE", "Release")
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF");

    if target.contains("windows") {
        cfg.cxxflag("/EHsc");
    }

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        cfg.define("LLAMA_CUDA", "ON")
            .define("GGML_CUDA", "ON")
            .define("CMAKE_CUDA_ARCHITECTURES", "89");
    }
    if env::var("CARGO_FEATURE_METAL").is_ok() && target.contains("apple") {
        cfg.define("LLAMA_METAL", "ON");
    }
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        cfg.define("LLAMA_VULKAN", "ON");
    }

    let dst = cfg.build();

    println!(
        "cargo:rustc-link-search=native={}/build/Release",
        dst.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/llama.cpp/src/Release",
        dst.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/llama.cpp/ggml/src/Release",
        dst.display()
    );
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!(
            "cargo:rustc-link-search=native={}/build/llama.cpp/ggml/src/ggml-cuda/Release",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    println!("cargo:rustc-link-lib=static=nezumi_llama_wrapper");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    apply_gpu_link(target);

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=native/llama_wrapper");
}

fn configure_litert_runtime() {
    let root = litert_lm_root();
    if !root.exists() {
        println!(
            "cargo:warning=LiteRT-LM not found at {}; clone into LiteRT-LM/ or set LITERT_LM_ROOT",
            root.display()
        );
        return;
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("LITERT_LM_MAIN") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(root.join("bazel-bin/runtime/engine/litert_lm_main.exe"));
    if let Some(er) = detect_bazel_execroot(&root) {
        candidates.push(
            er.join("bazel-out/x64_windows-opt/bin/runtime/engine/litert_lm_main.exe"),
        );
    }
    // Bazel --output_base=C:/bzl (see README)
    candidates.push(PathBuf::from(
        "C:/bzl/execroot/litert_lm/bazel-out/x64_windows-opt/bin/runtime/engine/litert_lm_main.exe",
    ));

    let mut found = false;
    for candidate in candidates {
        if candidate.exists() {
            println!(
                "cargo:rustc-env=LITERT_LM_MAIN={}",
                candidate.to_string_lossy()
            );
            found = true;
            break;
        }
    }
    if !found {
        println!(
            "cargo:warning=litert_lm_main.exe not found. Run scripts/build-litert-lm.ps1 or set LITERT_LM_MAIN."
        );
    }

    let prebuilt = root.join("prebuilt/windows_x86_64");
    if prebuilt.exists() {
        println!(
            "cargo:rustc-env=LITERT_LM_RUNTIME_DIR={}",
            prebuilt.to_string_lossy()
        );
    }

    println!("cargo:rerun-if-env-changed=LITERT_LM_ROOT");
    println!("cargo:rerun-if-env-changed=LITERT_LM_MAIN");
    println!("cargo:rerun-if-env-changed=LITERT_LM_RUNTIME_DIR");
}

fn apply_gpu_link(target: &str) {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/lib/x64");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }
    if env::var("CARGO_FEATURE_METAL").is_ok() && target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        println!("cargo:rustc-link-lib=vulkan");
    }
}
