use crate::{
    engines::selector::ModelFormat,
    engines::{ChatMessage, Engine, GenerateRequest, LoadConfig, ModelMeta},
    error::NezumiError,
};
use async_trait::async_trait;
use futures::Stream;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncReadExt, BufReader};
use tokio::process::Command;

const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(600);

/// LiteRT-LM inference via `litert_lm_main` subprocess (avoids linking the full Bazel graph).
pub struct LiteRTEngine {
    model_path: Mutex<Option<String>>,
    main_exe: PathBuf,
    runtime_dir: PathBuf,
}

impl LiteRTEngine {
    pub fn new() -> Self {
        Self {
            model_path: Mutex::new(None),
            main_exe: resolve_litert_main_exe(),
            runtime_dir: resolve_runtime_dir(),
        }
    }

    fn lock_model_path(&self) -> Result<std::sync::MutexGuard<'_, Option<String>>, NezumiError> {
        self.model_path
            .lock()
            .map_err(|e| NezumiError::ModelLoadFailed(format!("litert lock poisoned: {e}")))
    }
}

fn resolve_litert_main_exe() -> PathBuf {
    if let Ok(p) = std::env::var("LITERT_LM_MAIN") {
        return PathBuf::from(p);
    }
    if let Some(p) = option_env!("LITERT_LM_MAIN") {
        return PathBuf::from(p);
    }
    if let Ok(root) = std::env::var("LITERT_LM_ROOT") {
        let candidate = PathBuf::from(&root).join("bazel-bin/runtime/engine/litert_lm_main.exe");
        if candidate.exists() {
            return candidate;
        }
    }
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in [
        workspace.join("LiteRT-LM/bazel-bin/runtime/engine/litert_lm_main.exe"),
        PathBuf::from(
            "C:/bzl/execroot/litert_lm/bazel-out/x64_windows-opt/bin/runtime/engine/litert_lm_main.exe",
        ),
        workspace.join("LiteRT-LM/bazel-bin/runtime/engine/litert_lm_main"),
    ] {
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("litert_lm_main.exe")
}

fn resolve_runtime_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LITERT_LM_RUNTIME_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = option_env!("LITERT_LM_RUNTIME_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(root) = std::env::var("LITERT_LM_ROOT") {
        let prebuilt = PathBuf::from(&root).join("prebuilt/windows_x86_64");
        if prebuilt.exists() {
            return prebuilt;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("LiteRT-LM/prebuilt/windows_x86_64")
}

fn max_tokens_i32(max_tokens: Option<usize>) -> i32 {
    max_tokens
        .unwrap_or(512)
        .min(i32::MAX as usize) as i32
}

#[async_trait]
impl Engine for LiteRTEngine {
    fn supports(&self, meta: &ModelMeta) -> bool {
        matches!(meta.format, ModelFormat::TfLite)
    }

    async fn load(&self, path: &str, _config: LoadConfig) -> Result<(), NezumiError> {
        if !self.main_exe.exists() {
            return Err(NezumiError::EngineUnavailable(format!(
                "LiteRT-LM binary not found at {}. Build it with:\n  \
                 cd LiteRT-LM && bazelisk --output_base=C:/bzl build //runtime/engine:litert_lm_main --config=windows\n\
                 Then set LITERT_LM_MAIN or LITERT_LM_ROOT.",
                self.main_exe.display()
            )));
        }
        let lower = path.to_lowercase();
        if !lower.ends_with(".litertlm") && !lower.ends_with(".tflite") {
            return Err(NezumiError::UnsupportedModel(format!(
                "LiteRT-LM expects a .litertlm model file, got: {path}"
            )));
        }
        *self.lock_model_path()? = Some(path.to_string());
        Ok(())
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        let mut prompt = String::new();
        for msg in messages {
            if msg.role == "system" {
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
        }
        for msg in messages {
            match msg.role.as_str() {
                "user" => {
                    prompt.push_str("User: ");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                "assistant" | "model" => {
                    prompt.push_str("Assistant: ");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                _ => {}
            }
        }
        prompt.push_str("Assistant:");

        let req = GenerateRequest {
            prompt,
            max_tokens,
            temperature,
        };
        self.generate(req).await
    }

    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        let model_path = self
            .lock_model_path()?
            .clone()
            .ok_or_else(|| NezumiError::GenerationFailed("No LiteRT-LM model loaded".into()))?;

        let main_exe = self.main_exe.clone();
        let runtime_dir = self.runtime_dir.clone();
        let prompt = req.prompt;
        let input_prompt = if prompt.contains("<start_of_turn>") {
            prompt.clone()
        } else {
            format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                prompt
            )
        };
        let max_tokens = max_tokens_i32(req.max_tokens);
        let temperature = req.temperature.unwrap_or(0.8);

        Ok(run_litert_subprocess(
            &main_exe,
            &runtime_dir,
            &model_path,
            &input_prompt,
            max_tokens,
            temperature,
        )?)
    }
}

fn run_litert_subprocess(
    main_exe: &Path,
    runtime_dir: &Path,
    model_path: &str,
    input_prompt: &str,
    max_tokens: i32,
    temperature: f32,
) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
    let work_dir = if runtime_dir.exists() {
        runtime_dir
    } else {
        main_exe.parent().unwrap_or(Path::new("."))
    };

    let mut args: Vec<String> = Vec::new();
    args.push("--backend=cpu".to_string());
    args.push(format!("--model_path={model_path}"));
    args.push(format!("--max_tokens={max_tokens}"));
    args.push(format!("--temperature={temperature}"));

    let mut temp_prompt_path: Option<std::path::PathBuf> = None;
    if cfg!(windows) {
        let mut tmp = std::env::temp_dir();
        let fname = format!("nezumi_input_prompt_{}.txt", std::process::id());
        tmp.push(fname);
        if let Err(e) = std::fs::write(&tmp, input_prompt.as_bytes()) {
            return Err(NezumiError::GenerationFailed(format!(
                "Failed to write temp input prompt: {e}"
            )));
        }
        temp_prompt_path = Some(tmp.clone());
        args.push(format!("--input_prompt_file={}", tmp.display()));
    } else {
        args.push(format!("--input_prompt={input_prompt}"));
    }

    let mut cmd = Command::new(main_exe);
    cmd.current_dir(work_dir)
        .args(args.iter().map(|s| s.as_str()))
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(windows)]
    if work_dir.exists() {
        let work = work_dir.to_string_lossy();
        let path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{work};{path}"));
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| NezumiError::GenerationFailed(format!("Failed to spawn litert_lm_main: {e}")))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| NezumiError::GenerationFailed("No stdout from litert_lm_main".into()))?;
    let mut reader = BufReader::new(stdout);

    let temp_prompt_path = temp_prompt_path.clone();
    let stream = async_stream::stream! {
        let mut pending_bytes = Vec::new();
        let mut parse_buffer = String::new();
        let mut output_started = false;
        let mut chunk = [0u8; 2048];
        let deadline = tokio::time::Instant::now() + SUBPROCESS_TIMEOUT;

        loop {
            if tokio::time::Instant::now() >= deadline {
                let _ = child.kill().await;
                yield "[LiteRT error: subprocess timed out]".to_string();
                break;
            }

            let n = match tokio::time::timeout_at(deadline, reader.read(&mut chunk)).await {
                Ok(Ok(0)) => break,
                Ok(Ok(n)) => n,
                Ok(Err(e)) => {
                    yield format!("[LiteRT error: {e}]");
                    break;
                }
                Err(_) => {
                    let _ = child.kill().await;
                    yield "[LiteRT error: subprocess timed out]".to_string();
                    break;
                }
            };

            pending_bytes.extend_from_slice(&chunk[..n]);
            let decoded = if let Ok(s) = std::str::from_utf8(&pending_bytes) {
                let decoded = s.to_string();
                pending_bytes.clear();
                decoded
            } else {
                let err = std::str::from_utf8(&pending_bytes).unwrap_err();
                let valid_up_to = err.valid_up_to();
                let valid_str = if valid_up_to > 0 {
                    unsafe { std::str::from_utf8_unchecked(&pending_bytes[..valid_up_to]) }
                } else {
                    ""
                };
                let valid = valid_str.to_string();
                pending_bytes.drain(..valid_up_to);
                valid
            };

            parse_buffer.push_str(&decoded);

            if !output_started {
                let marker_pos = parse_buffer
                    .find("<start_of_turn>model")
                    .map(|i| (i, "<start_of_turn>model".len()))
                    .or_else(|| parse_buffer.rfind("\nAssistant:").map(|i| (i + 1, "Assistant:".len())));
                if let Some((idx, marker_len)) = marker_pos {
                    parse_buffer.drain(..idx + marker_len);
                    output_started = true;
                    parse_buffer = parse_buffer.trim_start_matches(|c: char| c == '\r' || c == '\n' || c.is_whitespace()).to_string();
                } else {
                    if parse_buffer.len() > 4096 {
                        let tail = parse_buffer.split_off(parse_buffer.len() - 4096);
                        parse_buffer = tail;
                    }
                    continue;
                }
            }

            if output_started {
                if let Some(pos) = parse_buffer.find("\nUser:") {
                    let out = parse_buffer[..pos].trim().to_string();
                    if !out.is_empty() {
                        yield out;
                    }
                    parse_buffer.clear();
                    break;
                }
                if let Some(pos) = parse_buffer.find("BenchmarkInfo").or_else(|| parse_buffer.find("benchmark_info")) {
                    let out = parse_buffer[..pos].trim().to_string();
                    if !out.is_empty() {
                        yield out;
                    }
                    parse_buffer.clear();
                    break;
                }

                if !parse_buffer.is_empty() {
                    let out = parse_buffer.clone();
                    parse_buffer.clear();
                    yield out;
                }
            }
        }

        if output_started && !parse_buffer.trim().is_empty() {
            yield parse_buffer.trim().to_string();
        }

        let _ = child.wait().await;
        if let Some(path) = temp_prompt_path {
            let _ = std::fs::remove_file(path);
        }
    };

    Ok(Box::pin(stream))
}
