use crate::{
    engines::selector::ModelFormat,
    engines::{Engine, GenerateRequest, LoadConfig, ModelMeta},
    error::NezumiError,
};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

#[cfg(feature = "litert")]
mod ffi {
    unsafe extern "C" {
        pub fn litert_load_model(path: *const std::ffi::c_char);
        pub fn litert_generate(prompt: *const std::ffi::c_char) -> *const std::ffi::c_char;
        pub fn litert_free();
    }
}

pub struct LiteRTEngine;

impl LiteRTEngine {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Engine for LiteRTEngine {
    fn supports(&self, meta: &ModelMeta) -> bool {
        matches!(meta.format, ModelFormat::TfLite)
    }

    async fn load(&self, path: &str, _config: LoadConfig) -> Result<(), NezumiError> {
        #[cfg(not(feature = "litert"))]
        return Err(NezumiError::EngineUnavailable(format!(
            "LiteRT support is disabled at compile time; rebuild with --features litert to load {}",
            path
        )));

        #[cfg(feature = "litert")]
        {
            let c_path = std::ffi::CString::new(path)
                .map_err(|e| NezumiError::ModelLoadFailed(e.to_string()))?;
            tokio::task::spawn_blocking(move || unsafe {
                ffi::litert_load_model(c_path.as_ptr());
            })
            .await
            .map_err(|e| NezumiError::ModelLoadFailed(e.to_string()))
        }
    }

    async fn generate(
        &self,
        _req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        #[cfg(not(feature = "litert"))]
        return Err(NezumiError::EngineUnavailable(
            "LiteRT support is disabled at compile time; rebuild with --features litert".into(),
        ));

        #[cfg(feature = "litert")]
        {
            let prompt = _req.prompt;
            let result = tokio::task::spawn_blocking(move || {
                let c_prompt = std::ffi::CString::new(prompt.as_str()).ok()?;
                let raw = unsafe { ffi::litert_generate(c_prompt.as_ptr()) };
                if raw.is_null() {
                    return None;
                }
                Some(unsafe { std::ffi::CStr::from_ptr(raw) }.to_string_lossy().into_owned())
            })
            .await
            .map_err(|e| NezumiError::GenerationFailed(e.to_string()))?
            .unwrap_or_default();

            let stream = async_stream::stream! { yield result; };
            Ok(Box::pin(stream))
        }
    }
}
