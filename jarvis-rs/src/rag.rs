//! RAG: full-text keyword search over markdown files, plus server-side
//! multi-turn conversation memory. Port of jarvis/rag.py — no behavior
//! changes, including the linear keyword-count scoring (no embeddings).

use crate::time_util::now_iso;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn sanitize(s: &str, extra_ok: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || extra_ok.contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Serialize, Clone)]
pub struct SearchResult {
    pub path: String,
    pub title: String,
    pub score: usize,
    pub snippet: String,
}

#[derive(Serialize, Clone)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

pub struct KnowledgeBase {
    pub root: PathBuf,
    // Append-only session files are the only thing that needs
    // cross-request mutual exclusion; everything else is plain fs I/O.
    write_lock: Mutex<()>,
}

impl KnowledgeBase {
    pub fn new(root: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&root)?;
        for sub in ["facts", "documents", "conversations", "tools"] {
            std::fs::create_dir_all(root.join(sub))?;
        }
        Ok(Self {
            root,
            write_lock: Mutex::new(()),
        })
    }

    pub fn all_files(&self) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = walkdir::WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
            .map(|e| e.path().to_path_buf())
            .collect();
        files.sort();
        files
    }

    pub fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
        let terms: Vec<String> = query.to_lowercase().split_whitespace().map(String::from).collect();
        let mut results = Vec::new();
        for fpath in self.all_files() {
            let text = match std::fs::read_to_string(&fpath) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let text_lower = text.to_lowercase();
            let score: usize = terms.iter().map(|t| text_lower.matches(t.as_str()).count()).sum();
            if score == 0 {
                continue;
            }
            let title = text
                .lines()
                .find(|l| l.starts_with("# "))
                .map(|l| l[2..].trim().to_string())
                .unwrap_or_default();
            let mut snippet = String::new();
            for term in &terms {
                if let Some(idx) = text_lower.find(term.as_str()) {
                    let start = idx.saturating_sub(60);
                    let end = (idx + 120).min(text.len());
                    // Byte-safe: snap to char boundaries since we're slicing by byte offset.
                    let start = floor_char_boundary(&text, start);
                    let end = ceil_char_boundary(&text, end);
                    snippet = text[start..end].replace('\n', " ").trim().to_string();
                    break;
                }
            }
            snippet.truncate(200);
            let rel_path = fpath
                .strip_prefix(&self.root)
                .unwrap_or(&fpath)
                .to_string_lossy()
                .to_string();
            results.push(SearchResult { path: rel_path, title, score, snippet });
        }
        results.sort_by(|a, b| b.score.cmp(&a.score));
        results.truncate(max_results);
        results
    }

    pub fn add_document(&self, filename: &str, content: &str) -> std::io::Result<String> {
        let doc_dir = self.root.join("documents");
        std::fs::create_dir_all(&doc_dir)?;
        let safe_name = sanitize(filename, "_-.");
        let (fpath, final_content) = if safe_name.ends_with(".md") {
            (doc_dir.join(&safe_name), content.to_string())
        } else {
            let fpath = doc_dir.join(format!("{safe_name}.md"));
            let header = format!(
                "---\ntype: document\ncreated: {}\nsource: upload\n---\n\n# {}\n\n",
                now_iso(),
                filename
            );
            (fpath, header + content)
        };
        std::fs::write(&fpath, final_content)?;
        Ok(fpath.strip_prefix(&self.root).unwrap_or(&fpath).to_string_lossy().to_string())
    }

    pub fn get_knowledge_context(&self, query: &str, max_results: usize) -> String {
        let results = self.search(query, max_results);
        if results.is_empty() {
            return String::new();
        }
        let mut parts = vec!["Here is relevant information from the knowledge base:".to_string()];
        for r in &results {
            parts.push(format!("\n--- {} ---", r.title));
            parts.push(r.snippet.clone());
        }
        parts.join("\n")
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        let mut safe = sanitize(session_id, "_-");
        safe.truncate(128);
        let safe = if safe.is_empty() { "default".to_string() } else { safe };
        self.root.join("conversations").join(format!("{safe}.md"))
    }

    pub fn save_turn(&self, session_id: &str, role: &str, content: &str) -> std::io::Result<()> {
        let _guard = self.write_lock.lock().unwrap();
        let fpath = self.session_path(session_id);
        let ts = now_iso();
        if !fpath.exists() {
            std::fs::write(
                &fpath,
                format!("---\ntype: conversation\nsession: {session_id}\ncreated: {ts}\n---\n\n# Session {session_id}\n\n"),
            )?;
        }
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&fpath)?;
        write!(f, "\n## {ts} — {role}\n\n{content}\n")?;
        Ok(())
    }

    pub fn get_recent_conversation(&self, session_id: &str, max_turns: usize) -> Vec<ConversationTurn> {
        let fpath = self.session_path(session_id);
        let text = match std::fs::read_to_string(&fpath) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut turns = Vec::new();
        // Mirrors Python's `text.split("\n## ")[1:]` — skip the preamble block.
        let mut blocks = text.split("\n## ");
        blocks.next();
        for block in blocks {
            let (header, body) = match block.split_once('\n') {
                Some((h, b)) => (h, b),
                None => (block, ""),
            };
            // header looks like "<ts> — <role>"
            let role = match header.rsplit_once(" — ") {
                Some((_, role)) => role.trim(),
                None => continue,
            };
            turns.push(ConversationTurn { role: role.to_string(), content: body.trim().to_string() });
        }
        let len = turns.len();
        if len > max_turns {
            turns.split_off(len - max_turns)
        } else {
            turns
        }
    }

    pub fn list_sessions(&self) -> Vec<String> {
        let conv_dir = self.root.join("conversations");
        let mut names: Vec<String> = std::fs::read_dir(&conv_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
            .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().to_string()))
            .collect();
        names.sort();
        names
    }
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

pub fn audit_log_path(kb_root: &Path) -> PathBuf {
    kb_root.join("tools").join("audit.log")
}
