use crate::engines::EngineType;

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub path: String,
    pub format: ModelFormat,
    pub context_len: Option<usize>,
    pub quantization: Quantization,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelFormat {
    Gguf,
    TfLite,
    Unknown,
}

/// モデルの量子化レベル（ファイル名ヒューリスティック）
#[derive(Debug, Clone, PartialEq)]
pub enum Quantization {
    High,   // Q8 / F16 / F32
    Medium, // Q4 / Q5 / Q6
    Low,    // Q2 / Q3
    Unknown,
}

impl ModelMeta {
    pub fn from_path(path: &str) -> Self {
        let lower = path.to_lowercase();
        let format = if lower.ends_with(".gguf") {
            ModelFormat::Gguf
        } else if lower.ends_with(".tflite") || lower.ends_with(".litertlm") {
            ModelFormat::TfLite
        } else {
            ModelFormat::Unknown
        };

        let quantization = if lower.contains("q8") || lower.contains("f16") || lower.contains("f32")
        {
            Quantization::High
        } else if lower.contains("q4") || lower.contains("q5") || lower.contains("q6") {
            Quantization::Medium
        } else if lower.contains("q2") || lower.contains("q3") {
            Quantization::Low
        } else {
            Quantization::Unknown
        };

        let context_len = parse_context_len_hint(&lower);

        Self {
            path: path.to_string(),
            format,
            context_len,
            quantization,
        }
    }
}

fn parse_context_len_hint(lower_path: &str) -> Option<usize> {
    for token in ["128k", "64k", "32k", "16k", "8k"] {
        if lower_path.contains(token) {
            let num: usize = token[..token.len() - 1].parse().ok()?;
            return Some(num * 1024);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub has_cuda: bool,
    pub has_metal: bool,
    pub has_vulkan: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        Self {
            has_cuda: cfg!(feature = "cuda") || std::env::var("NEZUMI_CUDA").is_ok(),
            has_metal: cfg!(feature = "metal") || std::env::var("NEZUMI_METAL").is_ok(),
            has_vulkan: cfg!(feature = "vulkan") || std::env::var("NEZUMI_VULKAN").is_ok(),
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.has_cuda || self.has_metal || self.has_vulkan
    }
}

#[derive(Debug, Clone, Default)]
pub enum UserPreference {
    #[default]
    Auto,
    SpeedFirst,
    QualityFirst,
}

pub struct EngineSelector;

impl EngineSelector {
    /// モデルメタ・ハードウェア・ユーザ設定から最適エンジンを選択
    pub fn select(meta: &ModelMeta, _hw: &HardwareProfile, _pref: &UserPreference) -> EngineType {
        match meta.format {
            ModelFormat::TfLite => EngineType::LiteRT,
            ModelFormat::Gguf | ModelFormat::Unknown => EngineType::Llama,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_without_gpu_uses_llama() {
        let meta = ModelMeta::from_path("gemma-3-1b-it-q4_k_m.gguf");
        let hw = HardwareProfile {
            has_cuda: false,
            has_metal: false,
            has_vulkan: false,
        };
        let pref = UserPreference::Auto;

        assert_eq!(EngineSelector::select(&meta, &hw, &pref), EngineType::Llama);
    }

    #[test]
    fn parses_context_len_from_filename() {
        let meta = ModelMeta::from_path("model-32k-q4.gguf");
        assert_eq!(meta.context_len, Some(32 * 1024));
    }
}
