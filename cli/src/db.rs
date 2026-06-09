use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
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

pub fn models_dir() -> PathBuf {
    nezumi_dir().join("models")
}

pub fn load_db() -> ModelsDb {
    let path = models_path();
    if !path.exists() {
        return ModelsDb::default();
    }
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    match toml::from_str(&content) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "Warning: failed to parse {}: {e}. Using empty model list.",
                path.display()
            );
            ModelsDb::default()
        }
    }
}

pub fn copy_model_into_store(src: &std::path::Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file_name = src
        .file_name()
        .ok_or("invalid source path")?
        .to_owned();
    let model_dir = models_dir();
    std::fs::create_dir_all(&model_dir)?;

    let mut dst_path = model_dir.join(&file_name);
    if dst_path.exists() {
        let stem = src
            .file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("model"));
        let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
        let mut index = 1;
        loop {
            let candidate = if ext.is_empty() {
                model_dir.join(format!("{}-{}", stem.to_string_lossy(), index))
            } else {
                model_dir.join(format!("{}-{}.{}", stem.to_string_lossy(), index, ext))
            };
            if !candidate.exists() {
                dst_path = candidate;
                break;
            }
            index += 1;
        }
    }

    std::fs::copy(src, &dst_path)?;
    Ok(dst_path)
}

pub fn save_db(db: &ModelsDb) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(nezumi_dir())?;
    let content = toml::to_string_pretty(db)?;
    std::fs::write(models_path(), content)?;
    Ok(())
}

pub fn key_from_name(name: &str) -> String {
    let normalized: String = name
        .chars()
        .map(|c| match c {
            ' ' => "__sp__".to_string(),
            ':' => "__co__".to_string(),
            '/' => "__sl__".to_string(),
            '\\' => "__bs__".to_string(),
            c if c.is_ascii_alphanumeric() || c == '_' || c == '-' => c.to_string(),
            _ => "__".to_string(),
        })
        .collect();
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    format!("{normalized}_{:016x}", hasher.finish())
}
