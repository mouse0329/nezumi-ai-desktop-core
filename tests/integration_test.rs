#[cfg(test)]
mod tests {
    use nezumi_ai_core::engines::{Engine, GenerateRequest};
    use nezumi_ai_core::engines::llama::LlamaEngine;
    use futures::StreamExt;

    const MODEL_PATH: &str = "C:\\Users\\mouse\\Downloads\\gemma-3-1b-it-q4_k_m.gguf";

    #[tokio::test]
    async fn test_load_and_generate() {
        let engine = LlamaEngine::new();

        engine.load(MODEL_PATH, nezumi_ai_core::engines::LoadConfig::full_gpu()).await.expect("モデルロード失敗");

        let req = GenerateRequest {
            prompt: "Hello, who are you?".to_string(),
            max_tokens: Some(64),
            temperature: Some(0.7),
        };

        let mut stream = engine.generate(req).await.expect("generate失敗");
        let mut output = String::new();

        while let Some(token) = stream.next().await {
            print!("{}", token);
            output.push_str(&token);
        }
        println!();

        assert!(!output.is_empty(), "出力が空");
    }
}
