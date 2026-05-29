use std::env;
use std::path::Path;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    build_llama(&target);
    build_litert();
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
        // MSVC は自動リンク
        println!("cargo:rustc-link-lib=advapi32");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=native/llama_wrapper");
}

fn build_litert() {
    let src = Path::new("native/litert_wrapper");
    if !src.join("CMakeLists.txt").exists() {
        if src.join("litert_wrapper.cpp").exists() {
            cc::Build::new()
                .cpp(true)
                .file(src.join("litert_wrapper.cpp"))
                .compile("litert_wrapper");
        }
        return;
    }

    let dst = cmake::Config::new(src)
        .define("CMAKE_BUILD_TYPE", "Release")
        .profile("Release")
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=nezumi_litert_wrapper");
    println!("cargo:rerun-if-changed=native/litert_wrapper");
}

fn apply_gpu_link(target: &str) {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/lib/x64");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cublas");
    }
    if env::var("CARGO_FEATURE_METAL").is_ok() && target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        println!("cargo:rustc-link-lib=vulkan");
    }
}
