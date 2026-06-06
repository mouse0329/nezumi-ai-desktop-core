use crate::{
    engines::selector::ModelFormat,
    engines::{ChatMessage, Engine, GenerateRequest, LoadConfig, ModelMeta},
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

#[repr(C)]
struct NezumiChatMessage {
    role: *const c_char,
    content: *const c_char,
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

    fn nezumi_llama_chat(
        state: *mut NezumiLlamaState,
        messages: *const NezumiChatMessage,
        n_messages: usize,
        max_tokens: i32,
        temperature: f32,
        cb: NezumiTokenCallback,
        user_data: *mut c_void,
    ) -> c_int;

    fn nezumi_llama_free(state: *mut NezumiLlamaState);
}

pub struct LlamaEngine {
    state: Mutex<*mut NezumiLlamaState>,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(std::ptr::null_mut()),
        }
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

        let cprompt = CString::new(req.prompt)
            .map_err(|_| NezumiError::InferenceError("invalid prompt".into()))?;
        let max_tokens = req.max_tokens.unwrap_or(512) as i32;
        let temperature = req.temperature.unwrap_or(0.8);
        let state_addr = ptr as usize;
        let prompt_bytes = cprompt.into_bytes_with_nul();

        spawn_llama_stream(state_addr, move |state_ptr, cb, user_data| {
            let cprompt = unsafe { CStr::from_bytes_with_nul_unchecked(&prompt_bytes) };
            unsafe {
                nezumi_llama_generate(
                    state_ptr,
                    cprompt.as_ptr(),
                    max_tokens,
                    temperature,
                    cb,
                    user_data,
                )
            }
        })
        .await
    }

    async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError> {
        let ptr = *self.state.lock().unwrap();
        if ptr.is_null() {
            return Err(NezumiError::ModelNotLoaded);
        }

        let role_strings: Vec<CString> = messages
            .iter()
            .map(|m| {
                CString::new(m.role.as_str())
                    .map_err(|_| NezumiError::InferenceError("invalid chat role".into()))
            })
            .collect::<Result<_, _>>()?;
        let content_strings: Vec<CString> = messages
            .iter()
            .map(|m| {
                CString::new(m.content.as_str())
                    .map_err(|_| NezumiError::InferenceError("invalid chat content".into()))
            })
            .collect::<Result<_, _>>()?;

        let max_tokens = max_tokens.unwrap_or(512) as i32;
        let temperature = temperature.unwrap_or(0.8);
        let state_addr = ptr as usize;

        spawn_llama_stream(state_addr, move |state_ptr, cb, user_data| {
            let c_messages: Vec<NezumiChatMessage> = role_strings
                .iter()
                .zip(content_strings.iter())
                .map(|(role, content)| NezumiChatMessage {
                    role: role.as_ptr(),
                    content: content.as_ptr(),
                })
                .collect();

            unsafe {
                nezumi_llama_chat(
                    state_ptr,
                    c_messages.as_ptr(),
                    c_messages.len(),
                    max_tokens,
                    temperature,
                    cb,
                    user_data,
                )
            }
        })
        .await
    }
}

struct TokenCallbackState {
    tx: tokio::sync::mpsc::UnboundedSender<String>,
}

async fn spawn_llama_stream<F>(
    state_addr: usize,
    run: F,
) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, NezumiError>
where
    F: FnOnce(*mut NezumiLlamaState, NezumiTokenCallback, *mut c_void) -> c_int + Send + 'static,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    unsafe extern "C" fn token_cb(token: *const c_char, user_data: *mut c_void) -> c_int {
        let state = &*(user_data as *mut TokenCallbackState);
        let s = CStr::from_ptr(token).to_string_lossy();
        // Raw token debug output is intentionally suppressed.
        if state.tx.send(s.into_owned()).is_err() {
            return 1;
        }
        0
    }

    let cb_state = TokenCallbackState { tx };
    let cb_state_addr = Box::into_raw(Box::new(cb_state)) as usize;

    tokio::task::spawn_blocking(move || {
        let state_ptr = state_addr as *mut NezumiLlamaState;
        let cb_state_ptr = cb_state_addr as *mut TokenCallbackState;
        let ret = run(
            state_ptr,
            token_cb,
            cb_state_ptr as *mut c_void,
        );

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
