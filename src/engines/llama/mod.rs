use crate::{
    engines::selector::ModelFormat,
    engines::{Engine, GenerateRequest, LoadConfig, ModelMeta},
    error::NezumiError,
};
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_void},
    pin::Pin,
    sync::Mutex,
};

#[repr(C)]
struct NezumiLlamaState {
    _opaque: [u8; 0],
}

type NezumiTokenCallback =
    unsafe extern "C" fn(token: *const c_char, user_data: *mut c_void) -> c_int;

type NezumiProgressCallback = unsafe extern "C" fn(progress: f32, user_data: *mut c_void);

extern "C" {
    fn nezumi_llama_load(
        model_path: *const c_char,
        n_ctx: i32,
        n_gpu_layers: i32,
        progress_cb: Option<NezumiProgressCallback>,
        progress_user_data: *mut c_void,
    ) -> *mut NezumiLlamaState;

    fn nezumi_llama_generate(
        state: *mut NezumiLlamaState,
        prompt: *const c_char,
        max_tokens: i32,
        temperature: f32,
        cb: NezumiTokenCallback,
        user_data: *mut c_void,
    ) -> c_int;

    fn nezumi_llama_free(state: *mut NezumiLlamaState);
}

pub struct LlamaEngine {
    state: Mutex<*mut NezumiLlamaState>,
    model_path: Mutex<Option<String>>,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(std::ptr::null_mut()),
            model_path: Mutex::new(None),
        }
    }
}

fn prompt_for_model(prompt: &str, model_path: Option<&str>) -> String {
    let is_qwen = model_path
        .map(|path| path.to_lowercase().contains("qwen"))
        .unwrap_or(false);

    if !is_qwen {
        return prompt.to_string();
    }

    prompt
        .replace("<start_of_turn>system\n", "<|im_start|>system\n")
        .replace("<start_of_turn>user\n", "<|im_start|>user\n")
        .replace("<start_of_turn>model\n", "<|im_start|>assistant\n")
        .replace("<end_of_turn>\n", "<|im_end|>\n")
        .replace("<end_of_turn>", "<|im_end|>")
}

#[cfg(test)]
mod tests {
    use super::prompt_for_model;

    #[test]
    fn qwen_prompt_uses_chatml_roles() {
        let prompt = "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n";
        let formatted = prompt_for_model(prompt, Some("Qwen3.5-4B-Q4_K_M.gguf"));

        assert_eq!(
            formatted,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn non_qwen_prompt_keeps_existing_template() {
        let prompt = "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n";

        assert_eq!(prompt_for_model(prompt, Some("gemma-3.gguf")), prompt);
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        let ptr = *self.state.lock().unwrap();
        if !ptr.is_null() {
            unsafe { nezumi_llama_free(ptr) };
        }
    }
}

#[async_trait]
impl Engine for LlamaEngine {
    fn supports(&self, meta: &ModelMeta) -> bool {
        matches!(meta.format, ModelFormat::Gguf | ModelFormat::Unknown)
    }

    async fn load(&self, path: &str, config: LoadConfig) -> Result<(), NezumiError> {
        let cpath =
            CString::new(path).map_err(|_| NezumiError::ModelLoadFailed("invalid path".into()))?;

        unsafe extern "C" fn progress_cb(progress: f32, _user_data: *mut c_void) {
            let pct = (progress * 100.0) as u32;
            print!(
                "\r\x1b[2K{} {}% [{}{}]",
                "Loading",
                pct,
                "#".repeat((pct / 5) as usize),
                ".".repeat((20 - pct / 5) as usize),
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        let ptr = unsafe {
            nezumi_llama_load(
                cpath.as_ptr(),
                config.n_ctx,
                config.n_gpu_layers,
                Some(progress_cb),
                std::ptr::null_mut(),
            )
        };
        println!();
        if ptr.is_null() {
            return Err(NezumiError::ModelLoadFailed(path.to_string()));
        }

        let mut guard = self.state.lock().unwrap();
        if !guard.is_null() {
            unsafe { nezumi_llama_free(*guard) };
        }
        *guard = ptr;
        *self.model_path.lock().unwrap() = Some(path.to_string());
        Ok(())
    }

    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        let ptr = *self.state.lock().unwrap();
        if ptr.is_null() {
            return Err(NezumiError::ModelNotLoaded);
        }

        let model_path = self.model_path.lock().unwrap().clone();
        let prompt = prompt_for_model(&req.prompt, model_path.as_deref());
        let cprompt = CString::new(prompt)
            .map_err(|_| NezumiError::InferenceError("invalid prompt".into()))?;
        let max_tokens = req.max_tokens.unwrap_or(512) as i32;
        let temperature = req.temperature.unwrap_or(0.8);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        struct CallbackState {
            tx: tokio::sync::mpsc::UnboundedSender<String>,
            pending: String,
            started: bool,
            ended: bool,
        }

        unsafe extern "C" fn token_cb(token: *const c_char, user_data: *mut c_void) -> c_int {
            let state = &mut *(user_data as *mut CallbackState);
            if state.ended {
                return 0;
            }

            let s = CStr::from_ptr(token).to_string_lossy();
            state.pending.push_str(&s);

            const PREFIX: &str = "assistant\n";
            if !state.started {
                if state.pending.len() < PREFIX.len() && PREFIX.starts_with(&state.pending) {
                    return 0;
                }

                if state.pending.starts_with(PREFIX) {
                    state.pending.drain(..PREFIX.len());
                }

                state.started = true;
            }

            if !state.pending.is_empty() {
                if state.tx.send(state.pending.clone()).is_err() {
                    return 1;
                }
            }
            state.pending.clear();
            0
        }

        let state_addr = ptr as usize;
        let cb_state = CallbackState {
            tx,
            pending: String::new(),
            started: false,
            ended: false,
        };
        let cb_state_addr = Box::into_raw(Box::new(cb_state)) as usize;
        let prompt_bytes = cprompt.into_bytes_with_nul();

        tokio::task::spawn_blocking(move || {
            let state_ptr = state_addr as *mut NezumiLlamaState;
            let cb_state_ptr = cb_state_addr as *mut CallbackState;
            let cprompt = unsafe { CStr::from_bytes_with_nul_unchecked(&prompt_bytes) };
            let ret = unsafe {
                nezumi_llama_generate(
                    state_ptr,
                    cprompt.as_ptr(),
                    max_tokens,
                    temperature,
                    token_cb,
                    cb_state_ptr as *mut c_void,
                )
            };

            let cb_state = unsafe { Box::from_raw(cb_state_ptr) };
            if ret != 0 {
                let _ = cb_state.tx.send(format!("Error: llama error {}", ret));
            }
        });

        Ok(Box::pin(stream! {
            while let Some(token) = rx.recv().await {
                yield token;
            }
        }))
    }
}
