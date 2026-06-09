#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use nezumi_ai_core::engines::llama::LlamaEngine;
    use nezumi_ai_core::engines::{Engine, GenerateRequest};

    #[tokio::test]
    async fn test_load_and_generate() {
        let model_path = std::env::var("NEZUMI_TEST_MODEL").unwrap_or_else(|_| {
            r"C:\Users\mouse\.nezumi-ai\models\Qwen3.5-4B-Q4_K_M.gguf".to_string()
        });

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("Skipping integration test: model not found at {model_path}");
            return;
        }

        let engine = LlamaEngine::new();

        engine
            .load(
                &model_path,
                nezumi_ai_core::engines::LoadConfig::full_gpu(),
            )
            .await
            .expect("モデルロード失敗");

        let req = GenerateRequest {
            prompt: "Hello, who are you?".to_string(),
            max_tokens: Some(64),
            temperature: Some(0.7),
        };

        let mut stream = engine.generate(req).await.expect("generate失敗");
        let mut output = String::new();

        while let Some(token) = stream.next().await {
            print!("{token}");
            output.push_str(&token);
        }
        println!();

        assert!(!output.is_empty(), "出力が空");
    }
}
