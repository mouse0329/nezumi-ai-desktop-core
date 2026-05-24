use futures::StreamExt;
use nezumi_ai_core::{Config, LoadConfig, NezumiCore};
use std::io::{self, Write};

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
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("使い方: nezumi <model_path> [system_prompt]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let system_prompt = args.get(2).cloned();

    println!("モデルをロード中: {}", model_path);

    let config = Config {
        system_prompt,
        ..Default::default()
    };
    let mut core = NezumiCore::init(config).await?;
    core.load_model(model_path, LoadConfig::full_gpu()).await?;

    println!("準備完了。Ctrl+C で終了。\n");

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }

        print!("ai>  ");
        io::stdout().flush()?;

        let mut stream = core.chat(input).await?;
        let mut buffer = String::new();
        'chat: while let Some(token) = stream.next().await {
            buffer.push_str(&token);

            loop {
                if let Some(idx) = buffer.find('<') {
                    if idx > 0 {
                        print!("{}", &buffer[..idx]);
                        buffer.drain(..idx);
                        continue;
                    }

                    if buffer.starts_with("<start_of_turn>") {
                        if !consume_start_of_turn_tag(&mut buffer) {
                            break;
                        }
                        continue;
                    }

                    if buffer.starts_with("<end_of_turn>") {
                        break 'chat;
                    }

                    if consume_unknown_tag(&mut buffer) {
                        continue;
                    }

                    break;
                }

                if !buffer.is_empty() {
                    print!("{}", buffer);
                    buffer.clear();
                }
                break;
            }

            io::stdout().flush()?;
        }
        println!();

        core.session.add("user", input).await?;
    }
}
