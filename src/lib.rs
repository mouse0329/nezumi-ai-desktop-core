pub mod engines;
pub mod error;
pub mod ffi;
pub mod session;

use engines::{
    create_engine, ChatMessage, Engine, EngineSelector, EngineType, GenerateRequest, HardwareProfile,
    ModelMeta, UserPreference,
};
use error::NezumiError;
use futures::{Stream, StreamExt};
use session::{InMemoryStore, SessionStore};
use std::{pin::Pin, sync::Arc};

pub use engines::{
    GenerateRequest as Request, LoadConfig, ModelMeta as Meta, UserPreference as Preference,
};
pub use error::NezumiError as Error;

pub struct Config {
    pub db_path: Option<String>,
    pub preference: UserPreference,
    pub system_prompt: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: None,
            preference: UserPreference::Auto,
            system_prompt: None,
        }
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

    pub async fn load_model(
        &mut self,
        path: &str,
        config: engines::LoadConfig,
    ) -> Result<(), NezumiError> {
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

    fn build_chat_messages(
        &self,
        history: &[crate::session::Message],
        user_input: &str,
    ) -> Vec<ChatMessage> {
        let mut messages = Vec::with_capacity(history.len() + 2);
        if let Some(ref sys) = self.system_prompt {
            messages.push(ChatMessage::new("system", sys));
        }
        for msg in history {
            messages.push(ChatMessage::new(&msg.role, &msg.content));
        }
        messages.push(ChatMessage::new("user", user_input));
        messages
    }

    pub async fn chat(
        &self,
        user_input: &str,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        let history = self.session.history().await?;
        let messages = self.build_chat_messages(&history, user_input);
        self.engine.chat(&messages, None, None).await
    }

    /// チャット生成+履歴保存
    pub async fn chat_and_save(
        &mut self,
        user_input: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        // メッセージ構築は session.add("user") より先に行う
        // → history に user が混入すると二重挿入になるため
        let history = self.session.history().await?;
        let messages = self.build_chat_messages(&history, user_input);

        self.session.add("user", user_input).await?;

        let mut inner = self.engine.chat(&messages, max_tokens, temperature).await?;
        let session = Arc::clone(&self.session);

        Ok(Box::pin(async_stream::stream! {
            let mut assistant_output = String::new();
            while let Some(token) = inner.next().await {
                assistant_output.push_str(&token);
                yield token;
            }
            let clean_output = strip_template_tags(&assistant_output);
            let clean_output = clean_output.trim();
            if !clean_output.is_empty() {
                let _ = session.add("model", clean_output).await;
            }
        }))
    }
}

fn strip_template_tags(s: &str) -> String {
    let s = if let Some(idx) = s.find("<end_of_turn>") {
        &s[..idx]
    } else {
        s
    };
    let mut result = String::new();
    let mut rest = s;
    while let Some(idx) = rest.find("<start_of_turn>") {
        result.push_str(&rest[..idx]);
        if let Some(nl) = rest[idx..].find('\n') {
            rest = &rest[idx + nl + 1..];
        } else {
            rest = "";
            break;
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_template_tags_removes_tags_and_end_marker() {
        let raw = "<start_of_turn>model\nHello<end_of_turn>\n<start_of_turn>user\nIgnored";
        let cleaned = strip_template_tags(raw);
        assert_eq!(cleaned, "Hello");
    }
}
