use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Deserialize, Serialize, Default, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub gpu_layers: Option<i32>,
    pub n_ctx: Option<i32>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Deserialize, Serialize, Default)]
pub struct ModelsDb {
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
}

pub fn nezumi_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".nezumi-ai")
}

pub fn models_path() -> PathBuf {
    nezumi_dir().join("models.toml")
}

pub fn load_db() -> ModelsDb {
    let path = models_path();
    if !path.exists() {
        return ModelsDb::default();
    }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    toml::from_str(&content).unwrap_or_default()
}

pub fn save_db(db: &ModelsDb) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(nezumi_dir())?;
    let content = toml::to_string_pretty(db)?;
    std::fs::write(models_path(), content)?;
    Ok(())
}

pub fn key_from_name(name: &str) -> String {
    name.replace([':', '/', ' '], "_")
}
