use crate::error::NezumiError;
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn add(&self, role: &str, content: &str) -> Result<(), NezumiError>;
    async fn history(&self) -> Result<Vec<Message>, NezumiError>;
    async fn clear(&self) -> Result<(), NezumiError>;
}

// --- InMemory実装（feature = "session-memory"、またはsqlite未使用時のフォールバック） ---

pub use memory::InMemoryStore;

pub mod memory {
    use super::*;
    use std::sync::Mutex;

    pub struct InMemoryStore {
        messages: Mutex<Vec<Message>>,
    }

    impl InMemoryStore {
        pub fn new() -> Self {
            Self { messages: Mutex::new(Vec::new()) }
        }
    }

    #[async_trait]
    impl SessionStore for InMemoryStore {
        async fn add(&self, role: &str, content: &str) -> Result<(), NezumiError> {
            self.messages.lock().unwrap().push(Message {
                role: role.to_string(),
                content: content.to_string(),
            });
            Ok(())
        }

        async fn history(&self) -> Result<Vec<Message>, NezumiError> {
            Ok(self.messages.lock().unwrap().clone())
        }

        async fn clear(&self) -> Result<(), NezumiError> {
            self.messages.lock().unwrap().clear();
            Ok(())
        }
    }
}

// --- SQLite実装（feature = "session-sqlite"） ---

#[cfg(feature = "session-sqlite")]
pub mod sqlite {
    use super::*;
    use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};

    pub struct SqliteStore {
        pool: SqlitePool,
    }

    impl SqliteStore {
        pub async fn new(db_path: &str) -> Result<Self, NezumiError> {
            let pool = SqlitePoolOptions::new()
                .connect(&format!("sqlite://{}?mode=rwc", db_path))
                .await?;
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS messages (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    role    TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts      INTEGER NOT NULL DEFAULT (unixepoch())
                )",
            )
            .execute(&pool)
            .await?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl SessionStore for SqliteStore {
        async fn add(&self, role: &str, content: &str) -> Result<(), NezumiError> {
            sqlx::query("INSERT INTO messages (role, content) VALUES (?, ?)")
                .bind(role)
                .bind(content)
                .execute(&self.pool)
                .await?;
            Ok(())
        }

        async fn history(&self) -> Result<Vec<Message>, NezumiError> {
            let rows = sqlx::query_as::<_, (String, String)>(
                "SELECT role, content FROM messages ORDER BY id",
            )
            .fetch_all(&self.pool)
            .await?;
            Ok(rows.into_iter().map(|(role, content)| Message { role, content }).collect())
        }

        async fn clear(&self) -> Result<(), NezumiError> {
            sqlx::query("DELETE FROM messages").execute(&self.pool).await?;
            Ok(())
        }
    }
}
