pub mod engines;
pub mod error;
pub mod ffi;
pub mod session;

use engines::{
    create_engine, Engine, EngineSelector, EngineType, GenerateRequest,
    HardwareProfile, ModelMeta, UserPreference,
};
use error::NezumiError;
use session::{InMemoryStore, SessionStore};
use std::sync::Arc;

pub use engines::{GenerateRequest as Request, ModelMeta as Meta, UserPreference as Preference, LoadConfig};
pub use error::NezumiError as Error;

pub struct Config {
    pub db_path: Option<String>,
    pub preference: UserPreference,
    pub system_prompt: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self { db_path: None, preference: UserPreference::Auto, system_prompt: None }
    }
}

pub struct NezumiCore {
    engine: Box<dyn Engine>,
    pub session: Arc<dyn SessionStore>,
    preference: UserPreference,
    system_prompt: Option<String>,
}

impl NezumiCore {
    pub async fn init(config: Config) -> Result<Self, NezumiError> {
        let session = Self::build_session(&config).await?;
        let engine = create_engine(EngineType::Llama);
        Ok(Self {
            engine,
            session,
            preference: config.preference,
            system_prompt: config.system_prompt,
        })
    }

    async fn build_session(config: &Config) -> Result<Arc<dyn SessionStore>, NezumiError> {
        #[cfg(feature = "session-sqlite")]
        if let Some(ref path) = config.db_path {
            return Ok(Arc::new(session::sqlite::SqliteStore::new(path).await?));
        }
        Ok(Arc::new(InMemoryStore::new()))
    }

    pub async fn load_model(&mut self, path: &str, config: engines::LoadConfig) -> Result<(), NezumiError> {
        let meta = ModelMeta::from_path(path);
        let hw = HardwareProfile::detect();
        let engine_type = EngineSelector::select(&meta, &hw, &self.preference);
        let engine = create_engine(engine_type);
        if !engine.supports(&meta) {
            return Err(NezumiError::UnsupportedModel(path.to_string()));
        }
        engine.load(path, config).await?;
        self.engine = engine;
        Ok(())
    }

    /// 生プロンプトで生成（テンプレートなし）
    pub async fn generate(
        &self,
        prompt: &str,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        self.engine.generate(GenerateRequest::new(prompt)).await
    }

    /// チャット形式で生成（履歴+Gemma3テンプレート）
    pub async fn chat(
        &self,
        user_input: &str,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        let history = self.session.history().await?;

        let mut prompt = String::new();

        // システムプロンプト
        if let Some(ref sys) = self.system_prompt {
            prompt.push_str(&format!(
                "<start_of_turn>system\n{}<end_of_turn>\n",
                sys
            ));
        }

        // 履歴
        for msg in &history {
            prompt.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role, msg.content
            ));
        }

        // 今回のユーザー入力
        prompt.push_str(&format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            user_input
        ));

        self.engine.generate(GenerateRequest::new(prompt)).await
    }

    /// チャット生成+履歴保存
    pub async fn chat_and_save(
        &mut self,
        user_input: &str,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        self.session.add("user", user_input).await?;
        self.chat(user_input).await
    }
}
