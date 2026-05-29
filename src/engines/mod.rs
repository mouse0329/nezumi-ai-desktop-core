pub mod litert;
pub mod llama;
pub mod selector;

use crate::error::NezumiError;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

pub use selector::{EngineSelector, HardwareProfile, ModelMeta, UserPreference};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    Llama,
    LiteRT,
}

#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub n_gpu_layers: i32,
    pub n_ctx: i32,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            n_ctx: 2048,
        }
    }
}

impl LoadConfig {
    pub fn gpu(n_gpu_layers: i32) -> Self {
        Self {
            n_gpu_layers,
            n_ctx: 2048,
        }
    }
    pub fn full_gpu() -> Self {
        Self {
            n_gpu_layers: 999,
            n_ctx: 2048,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[async_trait]
pub trait Engine: Send + Sync {
    async fn load(&self, path: &str, config: LoadConfig) -> Result<(), NezumiError>;
    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError>;
    fn supports(&self, meta: &ModelMeta) -> bool;
}

pub fn create_engine(engine_type: EngineType) -> Box<dyn Engine> {
    match engine_type {
        EngineType::Llama => Box::new(llama::LlamaEngine::new()),
        EngineType::LiteRT => Box::new(litert::LiteRTEngine::new()),
    }
}
