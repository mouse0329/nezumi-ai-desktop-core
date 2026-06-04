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

    // LiteRT-LM SDKビルド: リポジトリ内のLiteRT-LMまたは環境変数から取得
    let root = env::var("LITERT_LM_ROOT")
        .unwrap_or_else(|_| {
            // デフォルトではリポジトリ内のLiteRT-LMを使用
            let workspace_root = env::var("CARGO_MANIFEST_DIR").unwrap();
            format!("{}/LiteRT-LM", workspace_root)
        });
    
    let root_path = Path::new(&root);
    if !root_path.exists() {
        println!("cargo:warning=LiteRT-LM root not found at {}; skipping LiteRT-LM wrapper build", root);
        return;
    }
    // If the Bazel execroot isn't provided, remind the user to run Bazel
    // with UTF-8 copt flags. If LITERT_LM_AUTO_BAZEL is set, attempt
    // to run Bazel automatically with the required flags.
    if env::var("LITERT_LM_BAZEL_EXECROOT").is_err() {
        println!("cargo:warning=LITERT_LM_BAZEL_EXECROOT not set; please run Bazel in the LiteRT-LM repo and set this to the execroot (e.g. C:\\bazel-cache\\<id>\\execroot\\litert_lm).\nRun example:\nbazel build //... --copt=/utf-8 --host_copt=/utf-8 --verbose_failures");

        if env::var("LITERT_LM_AUTO_BAZEL").is_ok() {
            let bazel_cmd = env::var("BAZEL").unwrap_or_else(|_| "bazel".to_string());
            println!("cargo:warning=Attempting to run '{}' in {} (this may take a while)", bazel_cmd, root);
            match std::process::Command::new(&bazel_cmd)
                .current_dir(&root)
                .arg("build")
                .arg("//...")
                .arg("--copt=/utf-8")
                .arg("--host_copt=/utf-8")
                .arg("--verbose_failures")
                .output()
            {
                Ok(output) => {
                    if !output.status.success() {
                        println!("cargo:warning=Bazel build failed (exit {}). Stderr:\n{}", output.status, String::from_utf8_lossy(&output.stderr));
                    } else {
                        println!("cargo:warning=Bazel build completed successfully");
                    }
                }
                Err(e) => {
                    println!("cargo:warning=Failed to run bazel: {}", e);
                }
            }
        }
    }
    let mut cmake_config = cmake::Config::new(src);
    cmake_config
        .define("CMAKE_BUILD_TYPE", "Release")
        .profile("Release")
        .define("LITERT_LM_ROOT", &root);

    if let Ok(bazel_execroot) = env::var("LITERT_LM_BAZEL_EXECROOT") {
        cmake_config.define("LITERT_LM_BAZEL_EXECROOT", &bazel_execroot);
        println!("cargo:warning=Using LITERT_LM_BAZEL_EXECROOT={}", bazel_execroot);
    }

    println!("cargo:rustc-link-search=native={}/lib", root);
    
    // Link LiteRT-LM engine
    if let Ok(bazel_execroot) = env::var("LITERT_LM_BAZEL_EXECROOT") {
        // Use the Bazel-built link object library which includes all dependencies
        let _engine_impl_lib = format!(
            "{}/bazel-out/x64_windows-opt/bin/runtime/core/engine_advanced_impl_cpu_only.lo.lib",
            bazel_execroot
        );
        println!("cargo:rustc-link-search=native={}/bazel-out/x64_windows-opt/bin/runtime/core", bazel_execroot);
        println!("cargo:rustc-link-lib=static=engine_advanced_impl_cpu_only.lo");
        
        // Also add the base-level engine and cpp libraries
        println!("cargo:rustc-link-search=native={}/bazel-out/x64_windows-opt/bin/c", bazel_execroot);
        println!("cargo:rustc-link-lib=static=engine_cpu");
    } else {
        // Fallback to just engine_cpu if Bazel execroot not provided
        println!("cargo:rustc-link-lib=engine_cpu");
    }

    // Some CMake projects (like our litert wrapper) may not define an
    // "install" target. Request a direct build of the wrapper target
    // instead of relying on the default "install" target to avoid
    // MSBuild errors when install.vcxproj is missing.
    cmake_config.out_dir(std::env::var("OUT_DIR").unwrap() + "/litert");
    cmake_config.build_target("litert_wrapper");
    let dst = cmake_config.build();

    println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
    println!("cargo:rustc-link-lib=static=litert_wrapper");
    println!("cargo:rerun-if-changed=native/litert_wrapper");
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


