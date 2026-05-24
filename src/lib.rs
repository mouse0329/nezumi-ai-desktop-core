pub mod engines;
pub mod error;
pub mod ffi;
pub mod session;

use engines::{
    create_engine, Engine, EngineSelector, EngineType, GenerateRequest, LoadConfig,
    HardwareProfile, ModelMeta, UserPreference,
};
use error::NezumiError;
use session::{InMemoryStore, SessionStore};
use std::sync::Arc;

pub use engines::{GenerateRequest as Request, ModelMeta as Meta, UserPreference as Preference};
pub use error::NezumiError as Error;

pub struct Config {
    pub db_path: Option<String>,
    pub preference: UserPreference,
}

impl Default for Config {
    fn default() -> Self {
        Self { db_path: None, preference: UserPreference::Auto }
    }
}

pub struct NezumiCore {
    engine: Box<dyn Engine>,
    pub session: Arc<dyn SessionStore>,
    preference: UserPreference,
}

impl NezumiCore {
    pub async fn init(config: Config) -> Result<Self, NezumiError> {
        let session = Self::build_session(&config).await?;
        let engine = create_engine(EngineType::Llama);
        Ok(Self { engine, session, preference: config.preference })
    }

    async fn build_session(config: &Config) -> Result<Arc<dyn SessionStore>, NezumiError> {
        #[cfg(feature = "session-sqlite")]
        if let Some(ref path) = config.db_path {
            return Ok(Arc::new(session::sqlite::SqliteStore::new(path).await?));
        }
        Ok(Arc::new(InMemoryStore::new()))
    }

    pub async fn load_model(&mut self, path: &str) -> Result<(), NezumiError> {
        let meta = ModelMeta::from_path(path);
        let hw = HardwareProfile::detect();
        let engine_type = EngineSelector::select(&meta, &hw, &self.preference);

        let engine = create_engine(engine_type);
        if !engine.supports(&meta) {
            return Err(NezumiError::UnsupportedModel(path.to_string()));
        }
        engine.load(path, LoadConfig::default()).await?;
        self.engine = engine;
        Ok(())
    }

    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        self.engine.generate(GenerateRequest::new(prompt)).await
    }
}
