use thiserror::Error;

#[derive(Debug, Error)]
pub enum NezumiError {
    #[error("model load failed: {0}")]
    ModelLoadFailed(String),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("engine unavailable: {0}")]
    EngineUnavailable(String),
    #[error("inference error: {0}")]
    InferenceError(String),
    #[error("model not loaded")]
    ModelNotLoaded,
    #[error("ffi error: {0}")]
    FfiError(String),
    #[cfg(feature = "session-sqlite")]
    #[error("db error: {0}")]
    Db(#[from] sqlx::Error),
}

/// FFI境界でpanicさせないためのResult→i32変換
pub fn to_ffi_code(r: &Result<(), NezumiError>) -> i32 {
    match r {
        Ok(_) => 0,
        Err(NezumiError::ModelLoadFailed(_)) => -1,
        Err(NezumiError::UnsupportedModel(_)) => -2,
        Err(NezumiError::EngineUnavailable(_)) => -3,
        Err(NezumiError::InferenceError(_)) => -4,
        Err(NezumiError::ModelNotLoaded) => -5,
        Err(NezumiError::FfiError(_)) => -6,
        #[cfg(feature = "session-sqlite")]
        Err(NezumiError::Db(_)) => -7,
    }
}
