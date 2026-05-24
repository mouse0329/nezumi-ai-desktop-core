# nezumi-ai-desktop-core

**High-Performance AI Inference Library** — Rust と C++ で構築された、デスクトップ（Windows/macOS/Linux）向けオフラインAI推論コア。

## 概要

`nezumi-ai-desktop-core` は、Android版 `nezumi-ai` の推論機能をデスクトップ向けに再設計・最適化した**ヘッドレス・ライブラリ**です。特定のUI（Web/Native/CLI）に依存せず、Rust の安全性と C++ の計算速度を活かして、デバイス上での完全オフライン推論を提供します。

### コア・コンセプト

* **UI不可知論**: CUI、GUI（Tauri/Qt）、あるいはサーバーバックエンドなど、あらゆるインターフェースに組み込み可能。
* **ダブル推論バックエンド**:
    * **llama.cpp**: 高精度な GGUF モデル（Gemma 3n E4B 等）に対応。
    * **LiteRT-LM (TFLite)**: 超軽量・高速な TFLite モデル（Gemma 3n E2B 等）に対応。
* **ネイティブ・パフォーマンス**: Rust から直接 C++ エンジンを FFI 制御し、デスクトップの CPU/GPU リソースを最大化。

---

## アーキテクチャ

```text
[ Application Layer ] (CLI / Desktop-GUI / Web-Tauri)
          ↓ (Lib API call)
+---------------------------------------------------+
|             nezumi-ai-desktop-core                |
|  +---------------------------------------------+  |
|  |             Inference Manager               |  |
|  | (Engine Switching / Session Mgmt / History)  |  |
|  +---------------------------------------------+  |
|          ↓                       ↓                |
|  [ llama.cpp Bridge ]    [ LiteRT-LM Bridge ]     |
+----------↓-----------------------↓----------------+
     (Native Engine)         (Native Engine)
```

---

## 主な機能

* **エンジン・オーケストレーション**: モデル形式に応じた最適なバックエンド（llama.cpp または LiteRT-LM）の自動選択。
* **ストリーミング・プロトコル**: Rust の `Stream` またはコールバックを用いたリアルタイムなトークン生成。
* **マルチプラットフォーム最適化**:
    * **Windows**: CUDA / AVX2 / AVX512
    * **macOS**: Metal (Apple Silicon) / Accelerate Framework
    * **Linux**: CUDA / ROCm / Vulkan
* **セッション・永続化**: `sqlx` (SQLite) を内蔵し、コア単体で会話コンテキストの保存・復元が可能。

---

## プロジェクト構成

```text
nezumi-ai-desktop-core/
├── Cargo.toml            # ライブラリ定義・依存関係
├── build.rs              # C++ エンジン (llama.cpp / LiteRT) のコンパイル設定
├── src/
│   ├── lib.rs            # 外部公開用 API ファサード
│   ├── engines/          # 推論バックエンド抽象化層
│   │   ├── llama/        # llama.cpp ブリッジ
│   │   └── litert/       # LiteRT-LM ブリッジ
│   ├── session/          # 会話コンテキスト・履歴管理
│   └── error.rs          # 統合エラーハンドリング
└── native/               # C++ ソースコード・SDK
    ├── llama_wrapper/
    └── litert_wrapper/
```

---

## ライブラリの利用例 (Rust)

```rust
use nezumi_ai_core::{NezumiCore, EngineType, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // コアの初期化
    let core = NezumiCore::init(Config::default()).await?;

    // モデルのロード (GGUF なら llama.cpp, TFLite なら LiteRT が自動選択される)
    core.load_model("path/to/gemma-3n-e4b.gguf").await?;

    // ストリーミング推論
    let mut stream = core.generate("こんにちは、自己紹介して。").await?;
    while let Some(token) = stream.next().await {
        print!("{}", token);
    }

    Ok(())
}
```

---

## ライセンス

* **Core Logic**: LGPL v3 &独自License
* **Native Engines**:
    * llama.cpp (MIT)
    * LiteRT-LM (Apache 2.0)
