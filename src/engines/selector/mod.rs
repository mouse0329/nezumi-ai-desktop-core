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

        Self {
            path: path.to_string(),
            format,
            context_len: None,
            quantization,
        }
    }
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
            has_cuda: cfg!(feature = "cuda"),
            has_metal: cfg!(feature = "metal"),
            has_vulkan: cfg!(feature = "vulkan"),
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
    ///
    /// 判定優先順位:
    ///   1. フォーマット（TFLite → 常にLiteRT）
    ///   2. ユーザ設定 QualityFirst → 常にLlama
    ///   3. GPU有無 + 量子化レベル + context_len でフォールバック判定
    pub fn select(meta: &ModelMeta, hw: &HardwareProfile, pref: &UserPreference) -> EngineType {
        match meta.format {
            ModelFormat::TfLite => return EngineType::LiteRT,
            ModelFormat::Unknown => return EngineType::Llama,
            ModelFormat::Gguf => {}
        }

        // QualityFirst は常にllama.cpp
        if matches!(pref, UserPreference::QualityFirst) {
            return EngineType::Llama;
        }

        // GPU無し の場合、TFLite 以外は llama.cpp を使う
        if !hw.has_gpu() {
            let is_heavy_context = meta.context_len.map_or(false, |c| c > 8192);
            let is_high_quant = matches!(meta.quantization, Quantization::High);
            let speed_first = matches!(pref, UserPreference::SpeedFirst);

            if matches!(meta.format, ModelFormat::TfLite)
                && (speed_first || (!is_heavy_context && !is_high_quant))
            {
                return EngineType::LiteRT;
            }
        }

        EngineType::Llama
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
}
