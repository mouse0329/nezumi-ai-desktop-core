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

## LiteRT-LM で推論する

LiteRT-LM は `.litertlm` モデルを CPU で実行します。コアは `litert_lm_main` をサブプロセスとして起動します（Bazel の巨大な静的リンクを避けるため）。

### 1. LiteRT-LM をビルド（初回のみ）

リポジトリ直下の `LiteRT-LM/` を使います。Windows では **ユーザーディレクトリに空白があると Bazel が失敗**するため、`output_user_root` を短いパスに指定してください。

```powershell
cd LiteRT-LM
# Git Bash のパス（Bazel が genrule で使用）
$bash = "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
bazelisk --output_user_root=C:/bzl-user --output_base=C:/bzl build //runtime/engine:litert_lm_main --config=windows --shell_executable=$bash
```

ビルド後、実行ファイルの例:

`C:/bzl/execroot/litert_lm/bazel-out/x64_windows-opt/bin/runtime/engine/litert_lm_main.exe`

### 2. nezumi をビルド

```powershell
cargo build --features litert
cargo build -p nezumi-ai-cli
```

`.cargo/config.toml` の `LITERT_LM_ROOT` が `LiteRT-LM` を指していることを確認してください。実行ファイルを別の場所に置いた場合は `LITERT_LM_MAIN` を設定します。

### 3. 推論

```powershell
# テスト用モデル（リポジトリ内）
$model = "LiteRT-LM/runtime/testdata/test_lm.litertlm"

cargo run --bin nezumiai -p nezumi-ai-cli -- import $model --name my-litert
cargo run --bin nezumiai -p nezumi-ai-cli -- run my-litert
```

`.litertlm` / `.tflite` をロードするとエンジンセレクタが LiteRT を選びます。本番用モデルは [LiteRT-LM の Supported Models](https://github.com/google-ai-edge/LiteRT-LM) から取得してください。

---

## ライブラリの利用例 (Rust)

```rust
use futures::StreamExt;
use nezumi_ai_core::{Config, LoadConfig, NezumiCore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut core = NezumiCore::init(Config::default()).await?;

    // GGUF → llama.cpp、.litertlm → LiteRT-LM が自動選択される
    core.load_model("path/to/model.litertlm", LoadConfig::default()).await?;

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
