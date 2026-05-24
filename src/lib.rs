pub mod engines;
pub mod error;
pub mod session;

use engines::{detect_engine, create_engine, Engine};
use error::CoreError;
use futures::Stream;
use std::pin::Pin;

pub use engines::EngineType;

pub struct Config {
    pub db_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Self { db_path: "nezumi.db".into() }
    }
}

pub struct NezumiCore {
    engine: Box<dyn Engine>,
    pub session: session::SessionManager,
}

impl NezumiCore {
    pub async fn init(config: Config) -> Result<Self, CoreError> {
        let session = session::SessionManager::new(&config.db_path).await?;
        let engine = create_engine(EngineType::Llama);
        Ok(Self { engine, session })
    }

    pub async fn load_model(&mut self, path: &str) -> Result<(), CoreError> {
        self.engine = create_engine(detect_engine(path));
        self.engine.load(path).await
    }

    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, CoreError> {
        self.engine.generate(prompt).await
    }
}
