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
       .define("BUILD_SHARED_LIBS", "OFF")
       .define("LLAMA_BUILD_TESTS", "OFF")
       .define("LLAMA_BUILD_EXAMPLES", "OFF")
       .define("LLAMA_BUILD_SERVER", "OFF");

    // cmake-rs は MSVC で /EHsc を渡さないため明示追加
    if target.contains("windows") {
        cfg.cxxflag("/EHsc");
    }

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        cfg.define("LLAMA_CUDA", "ON");
    }
    if env::var("CARGO_FEATURE_METAL").is_ok() && target.contains("apple") {
        cfg.define("LLAMA_METAL", "ON");
    }
    if env::var("CARGO_FEATURE_VULKAN").is_ok() {
        cfg.define("LLAMA_VULKAN", "ON");
    }

    let dst = cfg.build();

    // cmake-rs は <dst>/build/ にビルド成果物を置く
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/llama.cpp", dst.display());
    println!("cargo:rustc-link-lib=static=nezumi_llama_wrapper");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");

    apply_gpu_link(target);

    // C++ 標準ライブラリ
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("windows") {
        // MSVC は自動リンク
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=native/llama_wrapper");
}

fn build_litert() {
    let src = Path::new("native/litert_wrapper");
    if !src.join("CMakeLists.txt").exists() {
        // スタブのみ: CMakeLists.txt未整備
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
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=nezumi_litert_wrapper");
    println!("cargo:rerun-if-changed=native/litert_wrapper");
}

fn apply_gpu_link(target: &str) {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-lib=cuda");
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
