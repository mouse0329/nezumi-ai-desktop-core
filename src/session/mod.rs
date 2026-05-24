use crate::error::CoreError;
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};

pub struct SessionManager {
    pool: SqlitePool,
}

impl SessionManager {
    pub async fn new(db_path: &str) -> Result<Self, CoreError> {
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

    pub async fn add(&self, role: &str, content: &str) -> Result<(), CoreError> {
        sqlx::query("INSERT INTO messages (role, content) VALUES (?, ?)")
            .bind(role)
            .bind(content)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn history(&self) -> Result<Vec<(String, String)>, CoreError> {
        let rows = sqlx::query_as::<_, (String, String)>(
            "SELECT role, content FROM messages ORDER BY id",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }
}
