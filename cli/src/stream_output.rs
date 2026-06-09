/// ストリーミングトークンを表示用に整形する。
/// 不完全なタグはバッファに残し、完成したタグだけ処理する。
pub fn drain_displayable(buffer: &mut String, show_thinking: bool) -> String {
    let mut out = String::new();
    loop {
        if consume_start_of_turn_tag(buffer) {
            continue;
        }

        if buffer.starts_with("<think>") {
            if let Some(end) = buffer.find("</think>") {
                if show_thinking {
                    let think_content =
                        buffer["<think>".len()..end].trim();
                    if !think_content.is_empty() {
                        out.push_str(&format!("\x1b[2m[思考] {think_content}\x1b[0m\n"));
                    }
                }
                buffer.drain(..end + "</think>".len());
                continue;
            }
            break;
        }

        if let Some(pos) = buffer.find('<') {
            if pos > 0 {
                out.push_str(&buffer[..pos]);
                buffer.drain(..pos);
            }
            if consume_unknown_tag(buffer) {
                continue;
            }
            break;
        }

        out.push_str(buffer);
        buffer.clear();
        break;
    }
    out
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

fn consume_unknown_tag(buffer: &mut String) -> bool {
    if let Some(end) = buffer.find('>') {
        buffer.drain(..=end);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_thinking_when_hidden() {
        let mut buf = "<think>\nsecret\n</think>\nHi".to_string();
        let out = drain_displayable(&mut buf, false);
        assert_eq!(out.trim(), "Hi");
        assert!(buf.is_empty());
    }

    #[test]
    fn waits_for_incomplete_tag() {
        let mut buf = "Hello<redacted_think".to_string();
        let out = drain_displayable(&mut buf, false);
        assert_eq!(out, "Hello");
        assert_eq!(buf, "<redacted_think");
    }
}
