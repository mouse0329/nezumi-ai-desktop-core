mod db;
use db::{load_db, save_db, key_from_name, models_dir, ModelEntry};

use futures::StreamExt;
use nezumi_ai_core::{Config, LoadConfig, NezumiCore};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

#[cfg(target_os = "windows")]
fn enable_windows_utf8() {
    use windows_sys::Win32::System::Console::{SetConsoleOutputCP, SetConsoleCP};
    unsafe {
        let _ = SetConsoleOutputCP(65001);
        let _ = SetConsoleCP(65001);
    }
}

#[cfg(not(target_os = "windows"))]
fn enable_windows_utf8() {}

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            if let Some(val) = args.get(i + 1) {
                if !val.starts_with("--") {
                    map.insert(args[i].trim_start_matches('-').to_string(), val.clone());
                    i += 2;
                    continue;
                }
            }
            map.insert(
                args[i].trim_start_matches('-').to_string(),
                "true".to_string(),
            );
        }
        i += 1;
    }
    map
}

fn print_usage() {
    eprintln!("Usage: nezumiai <command> [options]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  import <path> [options]   register a model");
    eprintln!("  run <name> [options]      run a model");
    eprintln!("  list                      list models");
    eprintln!("  show <name>               show model config");
    eprintln!("  set <name> [options]      update model config");
    eprintln!("  remove <name>             remove a model");
    eprintln!();
    eprintln!("Options (import/run/set):");
    eprintln!("  --name <name>             model name (required for import)");
    eprintln!("  --gpu <layers>            GPU layers (999=all, 0=CPU)");
    eprintln!("  --ctx <size>              context size (default: 2048)");
    eprintln!("  --temp <float>            temperature (default: 0.8)");
    eprintln!("  --max-tokens <int>        max tokens (default: 512)");
    eprintln!("  --system <prompt>         system prompt");
    eprintln!("  --think                   enable thinking mode (Qwen3.5 etc.)");
}

fn cmd_import(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = args.first().ok_or("path required")?;
    let opts = parse_args(args);
    let name = opts.get("name").ok_or("--name required")?;
    let src_path = Path::new(path);
    let file_name = src_path.file_name().ok_or("invalid source path")?;
    let model_dir = models_dir();
    std::fs::create_dir_all(&model_dir)?;

    let mut dst_path = model_dir.join(file_name);
    if dst_path.exists() {
        let stem = src_path.file_stem().unwrap_or_else(|| std::ffi::OsStr::new("model"));
        let ext = src_path.extension().and_then(|e| e.to_str()).unwrap_or("");
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

    std::fs::copy(src_path, &dst_path)?;
    let model_path = dst_path.to_string_lossy().to_string();

    let mut db = load_db();
    let key = key_from_name(name);
    db.models.insert(key, ModelEntry {
        name: name.clone(),
        path: model_path.clone(),
        gpu_layers: opts.get("gpu").and_then(|v| v.parse().ok()),
        n_ctx: opts.get("ctx").and_then(|v| v.parse().ok()),
        system_prompt: opts.get("system").cloned(),
        temperature: opts.get("temp").and_then(|v| v.parse().ok()),
        max_tokens: opts.get("max-tokens").and_then(|v| v.parse().ok()),
    });
    save_db(&db)?;
    println!("Imported: {} -> {}", name, model_path);
    Ok(())
}

fn cmd_set(name: &str, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut db = load_db();
    let key = key_from_name(name);
    let entry = db
        .models
        .get_mut(&key)
        .ok_or_else(|| format!("Not found: {}", name))?;
    let opts = parse_args(args);
    if let Some(v) = opts.get("gpu") {
        entry.gpu_layers = v.parse().ok();
    }
    if let Some(v) = opts.get("ctx") {
        entry.n_ctx = v.parse().ok();
    }
    if let Some(v) = opts.get("temp") {
        entry.temperature = v.parse().ok();
    }
    if let Some(v) = opts.get("max-tokens") {
        entry.max_tokens = v.parse().ok();
    }
    if let Some(v) = opts.get("system") {
        entry.system_prompt = Some(v.clone());
    }
    save_db(&db)?;
    println!("Updated: {}", name);
    Ok(())
}

fn cmd_list() -> Result<(), Box<dyn std::error::Error>> {
    let db = load_db();
    if db.models.is_empty() {
        println!("No models registered.");
        return Ok(());
    }
    println!(
        "{:<20} {:>5} {:>6} {:>5}  {}",
        "NAME", "GPU", "CTX", "TEMP", "PATH"
    );
    println!("{}", "-".repeat(70));
    let mut entries: Vec<&ModelEntry> = db.models.values().collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    for m in entries {
        println!(
            "{:<20} {:>5} {:>6} {:>5}  {}",
            m.name,
            m.gpu_layers.map(|v| v.to_string()).unwrap_or("-".into()),
            m.n_ctx.map(|v| v.to_string()).unwrap_or("-".into()),
            m.temperature
                .map(|v| format!("{:.1}", v))
                .unwrap_or("-".into()),
            m.path,
        );
    }
    Ok(())
}

fn cmd_show(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let db = load_db();
    let key = key_from_name(name);
    let m = db
        .models
        .get(&key)
        .ok_or_else(|| format!("Not found: {}", name))?;
    println!("name:         {}", m.name);
    println!("path:         {}", m.path);
    println!(
        "gpu_layers:   {}",
        m.gpu_layers
            .map(|v| v.to_string())
            .unwrap_or("999 (default)".into())
    );
    println!(
        "n_ctx:        {}",
        m.n_ctx
            .map(|v| v.to_string())
            .unwrap_or("2048 (default)".into())
    );
    println!(
        "temperature:  {}",
        m.temperature
            .map(|v| format!("{:.1}", v))
            .unwrap_or("0.8 (default)".into())
    );
    println!(
        "max_tokens:   {}",
        m.max_tokens
            .map(|v| v.to_string())
            .unwrap_or("512 (default)".into())
    );
    println!(
        "system:       {}",
        m.system_prompt.as_deref().unwrap_or("(none)")
    );
    Ok(())
}

fn cmd_remove(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut db = load_db();
    let key = key_from_name(name);
    if db.models.remove(&key).is_some() {
        save_db(&db)?;
        println!("Removed: {}", name);
    } else {
        eprintln!("Not found: {}", name);
    }
    Ok(())
}

async fn cmd_run(name: &str, args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let db = load_db();
    let key = key_from_name(name);
    let entry = db
        .models
        .get(&key)
        .ok_or_else(|| format!("Model not found: {}", name))?;
    let opts = parse_args(args);
    let gpu_layers = opts
        .get("gpu")
        .and_then(|v| v.parse().ok())
        .or(entry.gpu_layers)
        .unwrap_or(999);
    let n_ctx = opts
        .get("ctx")
        .and_then(|v| v.parse().ok())
        .or(entry.n_ctx)
        .unwrap_or(2048);
    let max_tokens = opts
        .get("max-tokens")
        .and_then(|v| v.parse().ok())
        .or(entry.max_tokens)
        .unwrap_or(512);
    let temperature = opts
        .get("temp")
        .and_then(|v| v.parse().ok())
        .or(entry.temperature)
        .unwrap_or(0.8f32);
    let system_prompt = opts
        .get("system")
        .cloned()
        .or_else(|| entry.system_prompt.clone());
    let thinking = opts.contains_key("think");
    println!("Loading: {} ({})", name, entry.path);
    let load_config = LoadConfig {
        n_gpu_layers: gpu_layers,
        n_ctx,
    };
    let core_config = Config {
        system_prompt,
        thinking,
        ..Default::default()
    };
    let mut core = NezumiCore::init(core_config).await?;
    core.load_model(&entry.path, load_config).await?;
    println!("Ready. Type 'exit' to quit, Ctrl+C to interrupt generation.\n");
    chat_loop(&mut core, max_tokens, temperature).await
}

async fn chat_loop(core: &mut NezumiCore, max_tokens: usize, temperature: f32) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        print!("you> ");
        io::stdout().flush()?;
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                println!("\nExiting...");
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(e.into()),
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" || input == ":q" {
            println!("\nExiting...");
            return Ok(());
        }
        print!("ai>  ");
        io::stdout().flush()?;
        let mut stream = core.chat_and_save(input, Some(max_tokens), Some(temperature)).await?;
            let mut buffer = String::new();
            let mut done = false;
            let mut skip_until_double_newline = false;  // 思考セクションをスキップ中か
            
            while !done {
                // 次のトークンを受け取る。ストリーム終了なら残バッファを吐いて終わり
                tokio::select! {
                    res = stream.next() => {
                        match res {
                            Some(token) => buffer.push_str(&token),
                            None => {
                                done = true;
                            }
                        }
                    }
                    _ = tokio::signal::ctrl_c() => {
                        println!("\n[Interrupted by user]");
                        done = true;
                    }
                }
                
                // バッファ処理
                loop {
                    if skip_until_double_newline {
                        // 思考セクション内 → 「\n\n」を探す
                        if let Some(pos) = buffer.find("\n\n") {
                            // 見つかった → スキップ終了
                            skip_until_double_newline = false;
                            buffer.drain(..pos + 2);  // 「\n\n」も含めて削除
                            continue;
                        } else {
                            // 見つからない → バッファ全部スキップ待ち
                            buffer.clear();
                            break;
                        }
                    } else {
                        // 通常モード → 「Thinking Process:」を探す
                        if let Some(pos) = buffer.find("Thinking Process:") {
                            // 見つかった → その前のテキストを出力
                            if pos > 0 {
                                print!("{}", &buffer[..pos]);
                            }
                            // スキップモード開始
                            skip_until_double_newline = true;
                            buffer.drain(..pos);
                            continue;
                        } else {
                            // 見つからない → バッファ全部出力
                            print!("{}", buffer);
                            buffer.clear();
                            break;
                        }
                    }
                }
                io::stdout().flush()?;
            }
            // ストリーム終了後に残ったテキストを出力（<end_of_turn>より前など）
            if !buffer.is_empty() {
                // タグを除去して残テキストだけ出力
                let clean: String = buffer
                    .split('<')
                    .enumerate()
                    .filter_map(|(i, part)| {
                        if i == 0 {
                            Some(part.to_string())
                        } else if let Some(end) = part.find('>') {
                            Some(part[end + 1..].to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                if !clean.is_empty() {
                    print!("{}", clean);
                }
            }
            println!();
    }
}

fn consume_start_of_turn_tag(buffer: &mut String) -> bool {
    const PREFIXES: [&str; 3] = [
        "<start_of_turn>user\n",
        "<start_of_turn>model\n",
        "<start_of_turn>system\n",
    ];
    for prefix in PREFIXES {
        if buffer.starts_with(prefix) {
            buffer.drain(..prefix.len());
            return true;
        }
    }
    if buffer.starts_with("<start_of_turn>") {
        if let Some(newline) = buffer.find('\n') {
            buffer.drain(..newline + 1);
            return true;
        }
    }
    false
}

fn consume_think_tag(buffer: &mut String) -> bool {
    if buffer.starts_with("<think>") {
        // Check if we have the complete closing tag
        if let Some(end) = buffer.find("</think>") {
            // Extract the thinking content (between tags)
            let think_content = &buffer["<think>".len()..end].trim();
            
            if !think_content.is_empty() {
                // Print thinking content with faint/dimmed style
                println!("\x1b[2m[思考] {}\x1b[0m", think_content);
            }
            
            // Remove both opening and closing tags and everything between
            buffer.drain(..end + "</think>".len());
            return true;
        }
        // Tag is not yet complete, wait for more tokens
        return false;
    }
    false
}

fn consume_unknown_tag(buffer: &mut String) -> bool {
    if let Some(end) = buffer.find('>') {
        buffer.drain(..=end);
        true
    } else {
        false
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_windows_utf8();

    // Ctrl+C でプロセスが終了しないようにハンドラを設定（何もしない）
    // これにより tokio::signal::ctrl_c() で制御可能になる
    let _ = ctrlc::set_handler(|| {
        // ここでは何もしない。chat_loop 内の tokio::select! で処理する。
    });

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("import") => cmd_import(&args[2..])?,
        Some("list") => cmd_list()?,
        Some("show") => {
            let name = args.get(2).ok_or("name required")?;
            cmd_show(name)?;
        }
        Some("set") => {
            let name = args.get(2).ok_or("name required")?;
            cmd_set(name, &args[3..])?;
        }
        Some("remove") => {
            let name = args.get(2).ok_or("name required")?;
            cmd_remove(name)?;
        }
        Some("run") => {
            let name = args.get(2).ok_or("model name required")?;
            cmd_run(name, &args[3..]).await?;
        }
        _ => print_usage(),
    }
    Ok(())
}
