fn main() {
    // llama_wrapper
    if std::path::Path::new("native/llama_wrapper/llama_wrapper.cpp").exists() {
        cc::Build::new()
            .cpp(true)
            .file("native/llama_wrapper/llama_wrapper.cpp")
            .compile("llama_wrapper");
    }

    // litert_wrapper
    if std::path::Path::new("native/litert_wrapper/litert_wrapper.cpp").exists() {
        cc::Build::new()
            .cpp(true)
            .file("native/litert_wrapper/litert_wrapper.cpp")
            .compile("litert_wrapper");
    }
}
