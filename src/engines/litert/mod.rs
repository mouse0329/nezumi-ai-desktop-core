use crate::{
    engines::selector::ModelFormat,
    engines::{Engine, GenerateRequest, LoadConfig, ModelMeta},
    error::NezumiError,
};
#[cfg(feature = "litert")]
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
#[cfg(feature = "litert")]
use litertlm::{Backend, Engine as LiteRtLmEngine, EngineSettings, SamplerParams};
use std::pin::Pin;
#[cfg(feature = "litert")]
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub struct LiteRTEngine {
    #[cfg(feature = "litert")]
    engine: Mutex<Option<Arc<LiteRtLmEngine>>>,
}

impl LiteRTEngine {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "litert")]
            engine: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Engine for LiteRTEngine {
    fn supports(&self, meta: &ModelMeta) -> bool {
        matches!(meta.format, ModelFormat::TfLite)
    }

    async fn load(&self, path: &str, _config: LoadConfig) -> Result<(), NezumiError> {
        #[cfg(not(feature = "litert"))]
        {
            return Err(NezumiError::EngineUnavailable(format!(
                "LiteRT support is disabled at compile time; rebuild with --features litert to load {}",
                path
            )));
        }

        #[cfg(feature = "litert")]
        {
            let config = _config;
            let model_path = PathBuf::from(path);
            let backend = if config.n_gpu_layers > 0 {
                Backend::Gpu
            } else {
                Backend::Cpu
            };
            let max_num_tokens = if config.n_ctx > 0 { config.n_ctx } else { 2048 };
            let cache_dir = std::env::temp_dir().join("nezumi-litert-cache");
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| NezumiError::ModelLoadFailed(format!("{}: {}", path, e)))?;

            let loaded = tokio::task::spawn_blocking(move || {
                LiteRtLmEngine::new(
                    EngineSettings::new(model_path)
                        .backend(backend)
                        .max_num_tokens(max_num_tokens)
                        .cache_dir(cache_dir),
                )
            })
            .await
            .map_err(|e| NezumiError::ModelLoadFailed(format!("{}: {}", path, e)))?
            .map_err(|e| NezumiError::ModelLoadFailed(format!("{}: {}", path, e)))?;

            let mut guard = self.engine.lock().unwrap();
            *guard = Some(Arc::new(loaded));
            Ok(())
        }
    }

    async fn generate(
        &self,
        _req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        #[cfg(not(feature = "litert"))]
        {
            return Err(NezumiError::EngineUnavailable(
                "LiteRT support is disabled at compile time; rebuild with --features litert".into(),
            ));
        }

        #[cfg(feature = "litert")]
        {
            let req = _req;
            let engine = self
                .engine
                .lock()
                .unwrap()
                .as_ref()
                .cloned()
                .ok_or(NezumiError::ModelNotLoaded)?;

            let temperature = req.temperature.unwrap_or(0.8);
            let sampler = if temperature <= 0.0 {
                SamplerParams::default().greedy()
            } else {
                SamplerParams::default()
                    .top_p(0.95)
                    .temperature(temperature)
            };
            let prompt = req.prompt;
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

            tokio::task::spawn_blocking(move || {
                let result = engine.create_session(sampler).and_then(|mut session| {
                    session.generate_stream(&prompt, |token| tx.send(token.to_string()).is_ok())
                });

                if let Err(err) = result {
                    let _ = tx.send(format!("Error: litert error {}", err));
                }
            });

            Ok(Box::pin(stream! {
                while let Some(token) = rx.recv().await {
                    yield token;
                }
            }))
        }
    }
}
