use crate::{engines::Engine, error::CoreError};
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
    async fn load(&self, _path: &str) -> Result<(), CoreError> {
        // TODO: FFI call to native/litert_wrapper
        Ok(())
    }

    async fn generate(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, CoreError> {
        let prompt = prompt.to_string();
        Ok(Box::pin(stream! {
            // TODO: FFI streaming from LiteRT
            yield format!("[litert] {}", prompt);
        }))
    }
}
