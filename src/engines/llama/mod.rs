use crate::{
    engines::{Engine, GenerateRequest, LoadConfig, ModelMeta},
    engines::selector::ModelFormat,
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

type NezumiProgressCallback =
    unsafe extern "C" fn(progress: f32, user_data: *mut c_void);

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
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    pub fn new() -> Self {
        Self { state: Mutex::new(std::ptr::null_mut()) }
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
        let cpath = CString::new(path)
            .map_err(|_| NezumiError::ModelLoadFailed("invalid path".into()))?;

        unsafe extern "C" fn progress_cb(progress: f32, _user_data: *mut c_void) {
            let pct = (progress * 100.0) as u32;
            print!("\r\x1b[2K{} {}% [{}{}]",
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

        let cprompt = CString::new(req.prompt.clone())
            .map_err(|_| NezumiError::InferenceError("invalid prompt".into()))?;
        let max_tokens = req.max_tokens.unwrap_or(512) as i32;
        let temperature = req.temperature.unwrap_or(0.8);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        unsafe extern "C" fn token_cb(
            token: *const c_char,
            user_data: *mut c_void,
        ) -> c_int {
            let tx = &*(user_data as *const tokio::sync::mpsc::UnboundedSender<String>);
            let s = CStr::from_ptr(token).to_string_lossy().into_owned();
            if tx.send(s).is_err() { 1 } else { 0 }
        }

        let state_addr = ptr as usize;
        let tx_addr = Box::into_raw(Box::new(tx)) as usize;
        let prompt_bytes = cprompt.into_bytes_with_nul();

        tokio::task::spawn_blocking(move || {
            let state_ptr = state_addr as *mut NezumiLlamaState;
            let tx_ptr = tx_addr as *mut tokio::sync::mpsc::UnboundedSender<String>;
            let cprompt = unsafe { CStr::from_bytes_with_nul_unchecked(&prompt_bytes) };
            let ret = unsafe {
                nezumi_llama_generate(
                    state_ptr,
                    cprompt.as_ptr(),
                    max_tokens,
                    temperature,
                    token_cb,
                    tx_ptr as *mut c_void,
                )
            };

            let tx = unsafe { Box::from_raw(tx_ptr) };
            if ret != 0 {
                let _ = tx.send(format!("Error: llama error {}", ret));
            }
        });

        Ok(Box::pin(stream! {
            while let Some(token) = rx.recv().await {
                yield token;
            }
        }))
    }
}
