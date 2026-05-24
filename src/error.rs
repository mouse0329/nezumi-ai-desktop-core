use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("engine error: {0}")]
    Engine(String),
    #[error("model not loaded")]
    ModelNotLoaded,
    #[error("db error: {0}")]
    Db(#[from] sqlx::Error),
}
