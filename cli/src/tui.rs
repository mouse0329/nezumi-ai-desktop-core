/// nezumiai-tui — arrow-key driven terminal UI for managing nezumiai models.
///
/// Build:  cargo build --bin nezumiai-tui
/// Run:    ./target/debug/nezumiai-tui
mod db;
use db::{key_from_name, load_db, save_db, ModelEntry};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;

// ─── Palette ────────────────────────────────────────────────────────────────

const C_BG: Color     = Color::Rgb(16, 16, 24);
const C_PANEL: Color  = Color::Rgb(24, 24, 36);
const C_BORDER: Color = Color::Rgb(60, 60, 90);
const C_ACCENT: Color = Color::Rgb(120, 180, 255);
const C_YELLOW: Color = Color::Rgb(255, 210, 80);
const C_GREEN: Color  = Color::Rgb(80, 220, 140);
const C_RED: Color    = Color::Rgb(255, 90, 90);
const C_DIM: Color    = Color::Rgb(100, 100, 130);
const C_WHITE: Color  = Color::Rgb(220, 220, 240);
const C_SEL_BG: Color = Color::Rgb(40, 60, 100);
const C_USER: Color   = Color::Rgb(120, 200, 255);
const C_AI: Color     = Color::Rgb(140, 230, 160);

// ─── Step sizes ─────────────────────────────────────────────────────────────
// Normal ← →: coarse.  Shift + ← →: fine (±1 / ±0.01).

const STEP_GPU_COARSE:   i32 = 10;
const STEP_GPU_FINE:     i32 = 1;
const STEP_CTX_COARSE:   i32 = 256;
const STEP_CTX_FINE:     i32 = 128;
const STEP_TEMP_COARSE:  f32 = 0.1;
const STEP_TEMP_FINE:    f32 = 0.01;
const STEP_TOK_COARSE:   i32 = 128;
const STEP_TOK_FINE:     i32 = 16;

// ─── Field descriptors ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum FieldKind { Int, Float, Str }

struct FieldDef {
    key:       &'static str,
    label:     &'static str,
    kind:      FieldKind,
    hint:      &'static str,
    int_min:   i32,
    int_max:   i32,
    float_min: f32,
    float_max: f32,
}

const FIELDS: &[FieldDef] = &[
    FieldDef { key: "gpu",        label: "GPU Layers",    kind: FieldKind::Int,
               hint: "999=all GPU  0=CPU   ←→ ±10   Shift+←→ ±1",
               int_min: 0, int_max: 999, float_min: 0.0, float_max: 0.0 },
    FieldDef { key: "ctx",        label: "Context Size",  kind: FieldKind::Int,
               hint: "Token window   ←→ ±256   Shift+←→ ±128",
               int_min: 128, int_max: 131072, float_min: 0.0, float_max: 0.0 },
    FieldDef { key: "temp",       label: "Temperature",   kind: FieldKind::Float,
               hint: "0.0=strict  2.0=wild   ←→ ±0.10   Shift+←→ ±0.01",
               int_min: 0, int_max: 0, float_min: 0.0, float_max: 2.0 },
    FieldDef { key: "max_tokens", label: "Max Tokens",    kind: FieldKind::Int,
               hint: "Tokens per reply   ←→ ±128   Shift+←→ ±16",
               int_min: 16, int_max: 32768, float_min: 0.0, float_max: 0.0 },
    FieldDef { key: "system",     label: "System Prompt", kind: FieldKind::Str,
               hint: "Enter to edit",
               int_min: 0, int_max: 0, float_min: 0.0, float_max: 0.0 },
];

// ─── Chat message ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ChatMessage {
    role:    &'static str, // "you" | "ai"
    content: String,
}

// ─── App state ───────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum Screen {
    Manager, // model list + settings
    Chat,    // chat with loaded model
}

#[derive(PartialEq, Clone, Copy)]
enum Focus { List, Settings }

#[derive(PartialEq)]
enum Mode {
    Normal,
    Editing,
    ImportPath,
    ImportName,
    Confirm(ConfirmAction),
}

#[derive(PartialEq, Clone, Copy)]
enum ConfirmAction { Remove }

struct App {
    // DB
    db:           db::ModelsDb,
    model_names:  Vec<String>,
    list_state:   ListState,
    // Manager UI
    screen:       Screen,
    focus:        Focus,
    field_sel:    usize,
    mode:         Mode,
    edit_buf:     String,
    import_path:  String,
    log:          Vec<(String, Color)>,
    // Chat UI
    loaded_model: Option<String>,   // name of currently loaded model
    chat_history: Vec<ChatMessage>,
    chat_input:   String,
    chat_scroll:  usize,            // lines scrolled up from bottom
}

impl App {
    fn new() -> Self {
        let db = load_db();
        let mut names: Vec<String> = db.models.values().map(|e| e.name.clone()).collect();
        names.sort();
        let mut list_state = ListState::default();
        if !names.is_empty() { list_state.select(Some(0)); }
        Self {
            db,
            model_names: names,
            list_state,
            screen:       Screen::Manager,
            focus:        Focus::List,
            field_sel:    0,
            mode:         Mode::Normal,
            edit_buf:     String::new(),
            import_path:  String::new(),
            log: vec![
                ("nezumiai TUI ready".into(), C_ACCENT),
                ("↑↓ select model  Enter load & chat  Tab switch pane  i import  s save  q quit".into(), C_DIM),
            ],
            loaded_model: None,
            chat_history: Vec::new(),
            chat_input:   String::new(),
            chat_scroll:  0,
        }
    }

    fn selected_entry(&self) -> Option<&ModelEntry> {
        let name = self.model_names.get(self.list_state.selected()?)?;
        self.db.models.get(&key_from_name(name))
    }

    fn selected_entry_mut(&mut self) -> Option<&mut ModelEntry> {
        let name = self.model_names.get(self.list_state.selected()?)?.clone();
        self.db.models.get_mut(&key_from_name(&name))
    }

    fn push_log(&mut self, msg: impl Into<String>, color: Color) {
        self.log.push((msg.into(), color));
        if self.log.len() > 300 { self.log.remove(0); }
    }

    fn refresh_names(&mut self) {
        let sel_name = self.list_state.selected()
            .and_then(|i| self.model_names.get(i)).cloned();
        self.model_names = self.db.models.values().map(|e| e.name.clone()).collect();
        self.model_names.sort();
        if self.model_names.is_empty() {
            self.list_state.select(None);
        } else {
            let idx = sel_name
                .and_then(|n| self.model_names.iter().position(|x| *x == n))
                .unwrap_or(0);
            self.list_state.select(Some(idx.min(self.model_names.len() - 1)));
        }
    }

    fn save(&mut self) {
        match save_db(&self.db) {
            Ok(_)  => self.push_log("✓ Saved to models.toml", C_GREEN),
            Err(e) => self.push_log(format!("✗ Save failed: {e}"), C_RED),
        }
    }

    // Load selected model into chat screen (simulated — real integration
    // requires async engine wiring outside the TUI event loop).
    fn load_selected(&mut self) {
        if let Some(entry) = self.selected_entry() {
            let name = entry.name.clone();
            self.loaded_model = Some(name.clone());
            self.chat_history.clear();
            self.chat_scroll = 0;
            self.chat_input.clear();
            self.chat_history.push(ChatMessage {
                role:    "ai",
                content: format!("Model '{}' loaded. How can I help?", name),
            });
            self.screen = Screen::Chat;
        }
    }

    // Commit typed chat message — wires into NezumiCore in real use.
    // Here we echo back a placeholder so the UI is fully exercisable.
    fn send_chat(&mut self) {
        let input = self.chat_input.trim().to_string();
        if input.is_empty() { return; }
        self.chat_history.push(ChatMessage { role: "you", content: input.clone() });
        // Placeholder response — replace with actual engine call when integrating.
        self.chat_history.push(ChatMessage {
            role:    "ai",
            content: format!("[model response to: {}]", input),
        });
        self.chat_input.clear();
        self.chat_scroll = 0; // jump to bottom on new message
    }

    fn commit_edit(&mut self) {
        let buf = self.edit_buf.clone();
        let fd        = &FIELDS[self.field_sel];
        let kind      = fd.kind;
        let key       = fd.key;
        let int_min   = fd.int_min;
        let int_max   = fd.int_max;
        let float_min = fd.float_min;
        let float_max = fd.float_max;
        if let Some(entry) = self.selected_entry_mut() {
            match kind {
                FieldKind::Int => {
                    if let Ok(v) = buf.parse::<i32>() {
                        apply_int(entry, key, v.clamp(int_min, int_max));
                    }
                }
                FieldKind::Float => {
                    if let Ok(v) = buf.parse::<f32>() {
                        apply_float(entry, key, v.clamp(float_min, float_max));
                    }
                }
                FieldKind::Str => { apply_str(entry, key, buf); }
            }
        }
    }

    fn step_field(&mut self, dir: i32, fine: bool) {
        let fd        = &FIELDS[self.field_sel];
        let kind      = fd.kind;
        let key       = fd.key;
        let int_min   = fd.int_min;
        let int_max   = fd.int_max;
        let float_min = fd.float_min;
        let float_max = fd.float_max;

        let int_step: i32 = match (key, fine) {
            ("gpu", false) => STEP_GPU_COARSE,
            ("gpu", true)  => STEP_GPU_FINE,
            ("ctx", false) => STEP_CTX_COARSE,
            ("ctx", true)  => STEP_CTX_FINE,
            ("max_tokens", false) => STEP_TOK_COARSE,
            ("max_tokens", true)  => STEP_TOK_FINE,
            _ => 1,
        };
        let float_step: f32 = match (key, fine) {
            ("temp", false) => STEP_TEMP_COARSE,
            ("temp", true)  => STEP_TEMP_FINE,
            _ => 0.1,
        };

        if let Some(entry) = self.selected_entry_mut() {
            match kind {
                FieldKind::Int => {
                    let cur = get_int(entry, key);
                    apply_int(entry, key, (cur + dir * int_step).clamp(int_min, int_max));
                }
                FieldKind::Float => {
                    let cur = get_float(entry, key);
                    let new = ((cur + dir as f32 * float_step) * 1000.0).round() / 1000.0;
                    apply_float(entry, key, new.clamp(float_min, float_max));
                }
                FieldKind::Str => {}
            }
        }
    }
}

// ─── Field accessors ─────────────────────────────────────────────────────────

fn get_int(e: &ModelEntry, key: &str) -> i32 {
    match key {
        "gpu"        => e.gpu_layers.unwrap_or(999),
        "ctx"        => e.n_ctx.unwrap_or(2048),
        "max_tokens" => e.max_tokens.unwrap_or(512) as i32,
        _            => 0,
    }
}
fn apply_int(e: &mut ModelEntry, key: &str, v: i32) {
    match key {
        "gpu"        => e.gpu_layers = Some(v),
        "ctx"        => e.n_ctx      = Some(v),
        "max_tokens" => e.max_tokens = Some(v as usize),
        _            => {}
    }
}
fn get_float(e: &ModelEntry, key: &str) -> f32 {
    match key { "temp" => e.temperature.unwrap_or(0.8), _ => 0.0 }
}
fn apply_float(e: &mut ModelEntry, key: &str, v: f32) {
    if key == "temp" { e.temperature = Some(v); }
}
fn get_str(e: &ModelEntry, key: &str) -> String {
    match key { "system" => e.system_prompt.clone().unwrap_or_default(), _ => String::new() }
}
fn apply_str(e: &mut ModelEntry, key: &str, v: String) {
    if key == "system" { e.system_prompt = if v.is_empty() { None } else { Some(v) }; }
}
fn field_display(e: &ModelEntry, fd: &FieldDef) -> String {
    match fd.kind {
        FieldKind::Int   => get_int(e, fd.key).to_string(),
        FieldKind::Float => format!("{:.2}", get_float(e, fd.key)),
        FieldKind::Str   => {
            let s = get_str(e, fd.key);
            if s.is_empty() { "(none)".into() } else { s }
        }
    }
}

// ─── Top-level UI router ─────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App) {
    let area = f.size();
    f.render_widget(Block::default().style(Style::default().bg(C_BG)), area);

    match app.screen {
        Screen::Manager => ui_manager(f, app, area),
        Screen::Chat    => ui_chat(f, app, area),
    }
}

// ─── Manager screen ──────────────────────────────────────────────────────────

fn ui_manager(f: &mut Frame, app: &App, area: Rect) {
    let log_h = 8u16.min(area.height / 4);
    let vsplit = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(log_h)])
        .split(area);

    let list_w = 30u16.min(area.width / 3);
    let hsplit = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(list_w), Constraint::Min(0)])
        .split(vsplit[0]);

    draw_model_list(f, app, hsplit[0]);
    draw_settings(f, app, hsplit[1]);
    draw_log(f, app, vsplit[1]);

    match &app.mode {
        Mode::ImportPath => draw_input_dialog(f, "Import — Step 1/2", "Model file path:", &app.edit_buf, area),
        Mode::ImportName => draw_input_dialog(f, "Import — Step 2/2", "Model name:", &app.edit_buf, area),
        Mode::Confirm(ConfirmAction::Remove) => draw_confirm_dialog(f, "Remove this model?", area),
        _ => {}
    }
}

fn draw_model_list(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::List && app.mode == Mode::Normal;
    let bc = if focused { C_ACCENT } else { C_BORDER };

    let items: Vec<ListItem> = app.model_names.iter().map(|name| {
        let loaded = app.loaded_model.as_deref() == Some(name.as_str());
        let label = if loaded { format!(" ● {}", name) } else { format!("   {}", name) };
        let color  = if loaded { C_GREEN } else { C_WHITE };
        ListItem::new(Line::from(Span::styled(label, Style::default().fg(color))))
    }).collect();

    let block = Block::default()
        .title(Span::styled(" MODELS ", Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(bc))
        .style(Style::default().bg(C_PANEL));

    let list = List::new(items).block(block)
        .highlight_style(Style::default().bg(C_SEL_BG).fg(C_YELLOW).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶");

    let mut state = app.list_state.clone();
    f.render_stateful_widget(list, area, &mut state);

    let hint_y = area.y + area.height.saturating_sub(2);
    if hint_y > area.y {
        f.render_widget(
            Paragraph::new(Span::styled(" Enter=chat  i=import  d=del", Style::default().fg(C_DIM))),
            Rect { x: area.x + 1, y: hint_y, width: area.width.saturating_sub(2), height: 1 },
        );
    }
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::Settings && app.mode == Mode::Normal;
    let editing = app.mode == Mode::Editing;
    let bc = if focused || editing { C_ACCENT } else { C_BORDER };

    let block = Block::default()
        .title(Span::styled(" SETTINGS ", Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(bc))
        .style(Style::default().bg(C_PANEL));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let Some(entry) = app.selected_entry() else {
        f.render_widget(
            Paragraph::new(Span::styled("  No model selected.", Style::default().fg(C_DIM))),
            inner,
        );
        return;
    };

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("  Model: ", Style::default().fg(C_DIM)),
            Span::styled(&*entry.name, Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Path:  ", Style::default().fg(C_DIM)),
            Span::styled(&*entry.path, Style::default().fg(C_DIM)),
        ]),
        Line::from(""),
    ];

    for (i, fd) in FIELDS.iter().enumerate() {
        let is_sel     = i == app.field_sel && app.focus == Focus::Settings;
        let is_editing = is_sel && app.mode == Mode::Editing;

        let label_style = if is_sel {
            Style::default().fg(C_YELLOW).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(C_ACCENT)
        };
        let prefix = if is_sel { "▶ " } else { "  " };

        let val_str = if is_editing {
            format!("{}█", app.edit_buf)
        } else {
            field_display(entry, fd)
        };
        let val_style = if is_editing {
            Style::default().fg(C_YELLOW).bg(C_SEL_BG).add_modifier(Modifier::BOLD)
        } else if is_sel {
            Style::default().fg(C_WHITE).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(C_WHITE)
        };

        lines.push(Line::from(vec![
            Span::styled(prefix, label_style),
            Span::styled(format!("{:<16}", fd.label), label_style),
            Span::styled(val_str, val_style),
        ]));
        if is_sel {
            lines.push(Line::from(
                Span::styled(format!("   {}", fd.hint), Style::default().fg(C_DIM))
            ));
        }
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "  s=save  Enter=load & chat  Tab=switch  q=quit",
        Style::default().fg(C_DIM),
    )));

    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(Span::styled(" LOG ", Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(C_BORDER))
        .style(Style::default().bg(C_PANEL));
    let inner = block.inner(area);
    let h = inner.height as usize;
    let start = app.log.len().saturating_sub(h);
    let lines: Vec<Line> = app.log[start..].iter()
        .map(|(msg, color)| Line::from(Span::styled(format!(" {msg}"), Style::default().fg(*color))))
        .collect();
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(lines), inner);
}

// ─── Chat screen ─────────────────────────────────────────────────────────────

fn ui_chat(f: &mut Frame, app: &App, area: Rect) {
    let input_h = 3u16;
    let vsplit = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(input_h)])
        .split(area);

    draw_chat_history(f, app, vsplit[0]);
    draw_chat_input(f, app, vsplit[1]);
}

fn draw_chat_history(f: &mut Frame, app: &App, area: Rect) {
    let model_name = app.loaded_model.as_deref().unwrap_or("model");
    let title = format!(" Chat: {} — Esc=back  PgUp/PgDn=scroll ", model_name);

    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(C_ACCENT))
        .style(Style::default().bg(C_PANEL));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let w = inner.width as usize;

    // Render all messages into wrapped lines first
    let mut all_lines: Vec<Line> = Vec::new();
    for msg in &app.chat_history {
        let (tag, color) = match msg.role {
            "you" => ("You", C_USER),
            _     => ("AI",  C_AI),
        };
        // header
        all_lines.push(Line::from(Span::styled(
            format!("┌─ {} ", tag),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        // word-wrap body
        let max_w = w.saturating_sub(4);
        let words = msg.content.split_whitespace().collect::<Vec<_>>();
        let mut line_buf = String::new();
        for word in words {
            if line_buf.is_empty() {
                line_buf.push_str(word);
            } else if line_buf.len() + 1 + word.len() <= max_w {
                line_buf.push(' ');
                line_buf.push_str(word);
            } else {
                all_lines.push(Line::from(Span::styled(
                    format!("│  {}", line_buf),
                    Style::default().fg(C_WHITE),
                )));
                line_buf = word.to_string();
            }
        }
        if !line_buf.is_empty() {
            all_lines.push(Line::from(Span::styled(
                format!("│  {}", line_buf),
                Style::default().fg(C_WHITE),
            )));
        }
        all_lines.push(Line::from(""));
    }

    let total = all_lines.len();
    let visible = inner.height as usize;
    let scroll_off = app.chat_scroll.min(total.saturating_sub(visible));
    let start = total.saturating_sub(visible).saturating_sub(scroll_off);
    let display: Vec<Line> = all_lines.into_iter().skip(start).take(visible).collect();

    f.render_widget(Paragraph::new(display), inner);
}

fn draw_chat_input(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(Span::styled(" Message — Enter=send  Esc=back ", Style::default().fg(C_DIM)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(C_BORDER))
        .style(Style::default().bg(C_PANEL));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let display = format!(" {}█", app.chat_input);
    f.render_widget(
        Paragraph::new(Span::styled(display, Style::default().fg(C_WHITE))),
        inner,
    );
}

// ─── Dialogs ─────────────────────────────────────────────────────────────────

fn draw_input_dialog(f: &mut Frame, title: &str, prompt: &str, buf: &str, area: Rect) {
    let dw = (area.width.saturating_sub(4)).min(70);
    let dh = 7u16;
    let dx = area.x + (area.width.saturating_sub(dw)) / 2;
    let dy = area.y + (area.height.saturating_sub(dh)) / 2;
    let da = Rect { x: dx, y: dy, width: dw, height: dh };
    f.render_widget(Clear, da);
    let block = Block::default()
        .title(Span::styled(format!(" {title} "), Style::default().fg(C_YELLOW).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(C_YELLOW))
        .style(Style::default().bg(C_PANEL));
    let inner = block.inner(da);
    f.render_widget(block, da);
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(format!(" {prompt}"), Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(format!(" {}█", buf), Style::default().fg(C_WHITE).bg(C_SEL_BG))),
        Line::from(""),
        Line::from(Span::styled(" Enter=confirm   Esc=cancel", Style::default().fg(C_DIM))),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_confirm_dialog(f: &mut Frame, title: &str, area: Rect) {
    let dw = 44u16.min(area.width.saturating_sub(4));
    let dh = 5u16;
    let dx = area.x + (area.width.saturating_sub(dw)) / 2;
    let dy = area.y + (area.height.saturating_sub(dh)) / 2;
    let da = Rect { x: dx, y: dy, width: dw, height: dh };
    f.render_widget(Clear, da);
    let block = Block::default()
        .title(Span::styled(format!(" {title} "), Style::default().fg(C_RED).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL).border_type(BorderType::Rounded)
        .border_style(Style::default().fg(C_RED))
        .style(Style::default().bg(C_PANEL));
    let inner = block.inner(da);
    f.render_widget(block, da);
    f.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled("  y = confirm   Esc / n = cancel", Style::default().fg(C_DIM))),
        ]).alignment(Alignment::Center),
        inner,
    );
}

// ─── Input handling ───────────────────────────────────────────────────────────

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    use KeyCode::*;

    // ── Chat screen ──────────────────────────────────────────────
    if app.screen == Screen::Chat {
        match key.code {
            Esc => { app.screen = Screen::Manager; }
            Enter => { app.send_chat(); }
            Backspace => { app.chat_input.pop(); }
            PageUp   => { app.chat_scroll = app.chat_scroll.saturating_add(5); }
            PageDown => { app.chat_scroll = app.chat_scroll.saturating_sub(5); }
            Up   => { app.chat_scroll = app.chat_scroll.saturating_add(1); }
            Down => { app.chat_scroll = app.chat_scroll.saturating_sub(1); }
            Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
            Char(c) => { app.chat_input.push(c); }
            _ => {}
        }
        return false;
    }

    // ── ImportPath dialog ────────────────────────────────────────
    if app.mode == Mode::ImportPath {
        match key.code {
            Esc       => { app.mode = Mode::Normal; app.edit_buf.clear(); }
            Backspace => { app.edit_buf.pop(); }
            Enter if !app.edit_buf.is_empty() => {
                app.import_path = app.edit_buf.clone();
                let stem = std::path::Path::new(&app.import_path)
                    .file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                app.edit_buf = stem;
                app.mode = Mode::ImportName;
            }
            Char(c) => { app.edit_buf.push(c); }
            _ => {}
        }
        return false;
    }

    // ── ImportName dialog ────────────────────────────────────────
    if app.mode == Mode::ImportName {
        match key.code {
            Esc       => { app.mode = Mode::Normal; app.edit_buf.clear(); app.import_path.clear(); }
            Backspace => { app.edit_buf.pop(); }
            Enter if !app.edit_buf.is_empty() => {
                let name = app.edit_buf.clone();
                let path = app.import_path.clone();
                app.db.models.insert(key_from_name(&name), ModelEntry {
                    name: name.clone(), path,
                    gpu_layers: None, n_ctx: None,
                    system_prompt: None, temperature: None, max_tokens: None,
                });
                app.refresh_names();
                if let Some(idx) = app.model_names.iter().position(|n| *n == name) {
                    app.list_state.select(Some(idx));
                }
                app.save();
                app.push_log(format!("✓ Imported '{name}'"), C_GREEN);
                app.mode = Mode::Normal;
                app.edit_buf.clear();
                app.import_path.clear();
            }
            Char(c) => { app.edit_buf.push(c); }
            _ => {}
        }
        return false;
    }

    // ── Confirm dialog ────────────────────────────────────────────
    if let Mode::Confirm(action) = app.mode {
        match key.code {
            Char('y') | Char('Y') => {
                if action == ConfirmAction::Remove {
                    if let Some(idx) = app.list_state.selected() {
                        if let Some(name) = app.model_names.get(idx).cloned() {
                            app.db.models.remove(&key_from_name(&name));
                            app.refresh_names();
                            app.save();
                            app.push_log(format!("✓ Removed '{name}'"), C_GREEN);
                        }
                    }
                }
                app.mode = Mode::Normal;
            }
            Esc | Char('n') | Char('N') => { app.mode = Mode::Normal; }
            _ => {}
        }
        return false;
    }

    // ── Inline edit ───────────────────────────────────────────────
    if app.mode == Mode::Editing {
        match key.code {
            Esc       => { app.mode = Mode::Normal; app.edit_buf.clear(); }
            Backspace => { app.edit_buf.pop(); }
            Enter => {
                app.commit_edit();
                app.mode = Mode::Normal;
                app.edit_buf.clear();
            }
            Char(c) => { app.edit_buf.push(c); }
            _ => {}
        }
        return false;
    }

    // ── Normal mode ───────────────────────────────────────────────
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    match key.code {
        Char('q') | Char('Q') => return true,
        Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,

        Tab => {
            app.focus = match app.focus {
                Focus::List     => Focus::Settings,
                Focus::Settings => Focus::List,
            };
        }

        Char('i') | Char('I') => { app.edit_buf.clear(); app.mode = Mode::ImportPath; }
        Char('s') | Char('S') => { app.save(); }
        Char('d') | Char('D') => {
            if app.selected_entry().is_some() { app.mode = Mode::Confirm(ConfirmAction::Remove); }
        }

        // List pane
        Up   | Char('k') if app.focus == Focus::List => {
            let i = app.list_state.selected().unwrap_or(0);
            if i > 0 { app.list_state.select(Some(i - 1)); }
        }
        Down | Char('j') if app.focus == Focus::List => {
            let i = app.list_state.selected().unwrap_or(0);
            if i + 1 < app.model_names.len() { app.list_state.select(Some(i + 1)); }
        }
        Enter if app.focus == Focus::List => { app.load_selected(); }

        // Settings pane — navigate fields
        Up   | Char('k') if app.focus == Focus::Settings => {
            if app.field_sel > 0 { app.field_sel -= 1; }
        }
        Down | Char('j') if app.focus == Focus::Settings => {
            if app.field_sel + 1 < FIELDS.len() { app.field_sel += 1; }
        }

        // Settings pane — adjust value
        Left  if app.focus == Focus::Settings => { app.step_field(-1, shift); }
        Right if app.focus == Focus::Settings => { app.step_field( 1, shift); }

        // Settings pane — type value
        Enter if app.focus == Focus::Settings => {
            if let Some(entry) = app.selected_entry() {
                let fd = &FIELDS[app.field_sel];
                app.edit_buf = field_display(entry, fd);
                if app.edit_buf == "(none)" { app.edit_buf.clear(); }
                app.mode = Mode::Editing;
            }
        }

        _ => {}
    }
    false
}

// ─── Main loop ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();
    let result = run_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    if let Err(e) = result { eprintln!("Error: {e}"); }
    Ok(())
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal.draw(|f| ui(f, app))?;
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if handle_key(app, key) { break; }
            }
        }
    }
    Ok(())
}