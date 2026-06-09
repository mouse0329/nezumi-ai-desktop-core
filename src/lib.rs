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
    pub thinking: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: None,
            preference: UserPreference::Auto,
            system_prompt: None,
            thinking: false,
        }
    }
}

pub struct NezumiCore {
    engine: Box<dyn Engine>,
    pub session: Arc<dyn SessionStore>,
    preference: UserPreference,
    system_prompt: Option<String>,
    thinking: bool,
    model_path: Option<String>,
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
            thinking: config.thinking,
            model_path: None,
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
        self.model_path = Some(path.to_string());
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
        let qwen_instant = self.should_use_qwen_instant_directive();

        if let Some(ref sys) = self.system_prompt {
            let sys_content = if !self.thinking && !qwen_instant {
                let mut sys_content = sys.clone();
                if !sys_content.contains("NO_THINK") && !sys_content.contains("THINK") {
                    if !sys_content.is_empty() {
                        sys_content.push(' ');
                    }
                    sys_content.push_str("NO_THINK");
                }
                sys_content
            } else {
                sys.clone()
            };
            messages.push(ChatMessage::new("system", sys_content));
        } else if !self.thinking && !qwen_instant {
            messages.push(ChatMessage::new("system", "NO_THINK"));
        } else if self.thinking {
            messages.push(ChatMessage::new(
                "system",
                "You are a helpful assistant. Think through the problem step by step before answering. Use your reasoning abilities to analyze the question carefully.",
            ));
        }

        for msg in history {
            messages.push(ChatMessage::new(&msg.role, &msg.content));
        }

        let user_content = if qwen_instant && !self.thinking {
            format!("{user_input}\n/no_think")
        } else {
            user_input.to_string()
        };
        messages.push(ChatMessage::new("user", &user_content));
        messages
    }

    fn should_use_qwen_instant_directive(&self) -> bool {
        if self.thinking {
            return false;
        }
        self.model_path
            .as_deref()
            .map(|path| path.to_lowercase().contains("qwen"))
            .unwrap_or(false)
    }

    pub fn thinking_enabled(&self) -> bool {
        self.thinking
    }

    pub async fn chat(
        &self,
        user_input: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<impl futures::Stream<Item = String>, NezumiError> {
        let history = self.session.history().await?;
        let messages = self.build_chat_messages(&history, user_input);
        self.engine.chat(&messages, max_tokens, temperature).await
    }

    /// 成功した会話ターンを履歴に保存する（user + assistant をセットで保存）
    pub async fn commit_chat_turn(
        &self,
        user_input: &str,
        assistant_output: &str,
    ) -> Result<(), NezumiError> {
        let clean = clean_assistant_output(assistant_output);
        if clean.is_empty() {
            return Ok(());
        }
        self.session.add("user", user_input).await?;
        self.session.add("model", &clean).await?;
        Ok(())
    }

    /// チャット生成。履歴保存は推論成功後に行う。
    pub async fn chat_and_save(
        &mut self,
        user_input: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        let history = self.session.history().await?;
        let messages = self.build_chat_messages(&history, user_input);
        let mut inner = self.engine.chat(&messages, max_tokens, temperature).await?;
        let session = Arc::clone(&self.session);
        let user_input = user_input.to_string();

        Ok(Box::pin(async_stream::stream! {
            let mut assistant_output = String::new();
            while let Some(token) = inner.next().await {
                assistant_output.push_str(&token);
                yield token;
            }
            let clean = clean_assistant_output(&assistant_output);
            if !clean.is_empty() {
                if let Err(e) = session.add("user", &user_input).await {
                    yield format!("[Session error: {e}]");
                } else if let Err(e) = session.add("model", &clean).await {
                    yield format!("[Session error: {e}]");
                }
            }
        }))
    }
}

pub fn clean_assistant_output(s: &str) -> String {
    strip_thinking_tags(&strip_template_tags(s))
        .trim()
        .to_string()
}

fn strip_thinking_tags(s: &str) -> String {
    let mut result = s.to_string();
    loop {
        let Some(start) = result.find("<think>") else {
            break;
        };
        if let Some(end) = result[start..].find("</think>") {
            let remove_end = start + end + "</think>".len();
            result.replace_range(start..remove_end, "");
        } else {
            result.truncate(start);
            break;
        }
    }
    result
}

fn strip_template_tags(s: &str) -> String {
    let s = if let Some(idx) = s.find("<end_of_turn>") {
        &s[..idx]
    } else if let Some(idx) = s.find("<|im_end|>") {
        &s[..idx]
    } else {
        s
    };
    let mut result = String::new();
    let mut rest = s;
    loop {
        let next = [rest.find("<start_of_turn>"), rest.find("<|im_start|>")]
            .into_iter()
            .flatten()
            .min();
        match next {
            Some(idx) => {
                result.push_str(&rest[..idx]);
                if let Some(nl) = rest[idx..].find('\n') {
                    rest = &rest[idx + nl + 1..];
                } else {
                    rest = "";
                    break;
                }
            }
            None => break,
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

    #[test]
    fn strip_thinking_tags_removes_thinking_block() {
        let raw = "<think>\nfoo\n</think>\nHello";
        let cleaned = strip_thinking_tags(raw);
        assert_eq!(cleaned.trim(), "Hello");
    }

    #[test]
    fn qwen_model_appends_no_think_to_last_user_message() {
        let core = NezumiCore {
            engine: create_engine(EngineType::Llama),
            session: Arc::new(InMemoryStore::new()),
            preference: UserPreference::Auto,
            system_prompt: None,
            thinking: false,
            model_path: Some("/models/Qwen3.5-4B-Q4_K_M.gguf".to_string()),
        };

        let messages = core.build_chat_messages(&[], "こんにちは");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "こんにちは\n/no_think");
    }
}
