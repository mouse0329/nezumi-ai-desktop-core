use crate::{
    engines::{Engine, GenerateRequest, ModelMeta},
    engines::selector::ModelFormat,
    error::NezumiError,
};
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

pub struct LiteRTEngine;

impl LiteRTEngine {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Engine for LiteRTEngine {
    fn supports(&self, meta: &ModelMeta) -> bool {
        matches!(meta.format, ModelFormat::TfLite)
    }

    async fn load(&self, _path: &str) -> Result<(), NezumiError> {
        // TODO: FFI call to native/litert_wrapper
        Ok(())
    }

    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        Ok(Box::pin(stream! {
            // TODO: FFI streaming from LiteRT
            yield format!("[litert] {}", req.prompt);
        }))
    }
}
