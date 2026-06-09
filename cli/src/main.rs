mod db;
mod stream_output;

use db::{copy_model_into_store, key_from_name, load_db, save_db, ModelEntry};
use futures::StreamExt;
use nezumi_ai_core::{Config, LoadConfig, NezumiCore};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use stream_output::drain_displayable;

#[cfg(target_os = "windows")]
fn enable_windows_utf8() {
    use windows_sys::Win32::System::Console::{SetConsoleCP, SetConsoleOutputCP};
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
    let model_path = copy_model_into_store(Path::new(path))?
        .to_string_lossy()
        .to_string();

    let mut db = load_db();
    let key = key_from_name(name);
    db.models.insert(
        key,
        ModelEntry {
            name: name.clone(),
            path: model_path.clone(),
            gpu_layers: opts.get("gpu").and_then(|v| v.parse().ok()),
            n_ctx: opts.get("ctx").and_then(|v| v.parse().ok()),
            system_prompt: opts.get("system").cloned(),
            temperature: opts.get("temp").and_then(|v| v.parse().ok()),
            max_tokens: opts.get("max-tokens").and_then(|v| v.parse().ok()),
        },
    );
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

async fn chat_loop(
    core: &mut NezumiCore,
    max_tokens: usize,
    temperature: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let show_thinking = core.thinking_enabled();
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

        let mut stream = core
            .chat_and_save(input, Some(max_tokens), Some(temperature))
            .await?;
        let mut buffer = String::new();
        let mut raw_output = String::new();
        let mut done = false;
        let mut interrupted = false;

        while !done {
            tokio::select! {
                res = stream.next() => {
                    match res {
                        Some(token) => {
                            raw_output.push_str(&token);
                            buffer.push_str(&token);
                            let display = drain_displayable(&mut buffer, show_thinking);
                            if !display.is_empty() {
                                print!("{display}");
                                io::stdout().flush()?;
                            }
                        }
                        None => done = true,
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    println!("\n[Interrupted by user]");
                    interrupted = true;
                    done = true;
                }
            }
        }

        if !buffer.is_empty() {
            let display = drain_displayable(&mut buffer, show_thinking);
            if !display.is_empty() {
                print!("{display}");
            }
        }

        if interrupted {
            core.commit_chat_turn(input, &raw_output).await?;
        }

        println!();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_windows_utf8();

    let _ = ctrlc::set_handler(|| {});

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
