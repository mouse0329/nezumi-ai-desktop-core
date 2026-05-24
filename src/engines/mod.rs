pub mod llama;
pub mod litert;

use crate::error::CoreError;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

pub enum EngineType {
    Llama,
    LiteRT,
}

#[async_trait]
pub trait Engine: Send + Sync {
    async fn load(&self, path: &str) -> Result<(), CoreError>;
    async fn generate(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, CoreError>;
}

pub fn create_engine(engine_type: EngineType) -> Box<dyn Engine> {
    match engine_type {
        EngineType::Llama => Box::new(llama::LlamaEngine::new()),
        EngineType::LiteRT => Box::new(litert::LiteRTEngine::new()),
    }
}

/// モデルパスの拡張子からエンジンを自動選択
pub fn detect_engine(model_path: &str) -> EngineType {
    if model_path.ends_with(".gguf") {
        EngineType::Llama
    } else {
        EngineType::LiteRT
    }
}
