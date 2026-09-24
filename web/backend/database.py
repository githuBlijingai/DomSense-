"""SQLite 数据库访问：反馈与训练记录的持久化。

使用标准库 sqlite3，check_same_thread=False，通过 threading.Lock 保护写操作。
"""
import os
import sqlite3
import threading

from .config import DB_PATH

# 全局写锁，保护所有 DB 写操作（SQLite 在多线程下需要序列化写）
_db_lock = threading.Lock()


def get_db_connection() -> sqlite3.Connection:
    """返回一个 SQLite 连接（check_same_thread=False，row_factory=Row）。

    注意：调用方需负责 close()。每次调用返回新连接，不缓存。
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """建表 feedbacks + training_records（如不存在）。同时确保目录存在。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state_text TEXT NOT NULL,
                actions TEXT NOT NULL,
                optimal_action INTEGER NOT NULL,
                transition_probs TEXT NOT NULL,
                expected_returns TEXT NOT NULL,
                predicted_choice INTEGER,
                source TEXT DEFAULT 'web',
                created_at TEXT NOT NULL,
                used_in_training INTEGER DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT,
                trigger_source TEXT,
                num_samples INTEGER,
                status TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                best_val_loss REAL,
                checkpoint_path TEXT,
                feedback_ids TEXT,
                log TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # 迁移：为 feedbacks 表增量添加 domain 列（老库无此列，需幂等补齐）
        cursor.execute("PRAGMA table_info(feedbacks)")
        cols = {r[1] for r in cursor.fetchall()}
        if "domain" not in cols:
            cursor.execute(
                "ALTER TABLE feedbacks ADD COLUMN domain TEXT DEFAULT 'general'"
            )
        conn.commit()
    finally:
        conn.close()


def execute_query(sql: str, params: tuple = ()):
    """通用查询：执行 SQL 并返回结果。

    - SELECT / WITH：返回 list[dict]（每行一个字典）
    - INSERT / UPDATE / DELETE：返回 [{"lastrowid": int, "changes": int}]

    所有操作通过 _db_lock 序列化，保证线程安全。

    Args:
        sql: SQL 语句
        params: 参数元组

    Returns:
        list[dict]
    """
    with _db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            stripped = sql.strip().upper()
            if stripped.startswith("SELECT") or stripped.startswith("WITH") or stripped.startswith("PRAGMA"):
                rows = cursor.fetchall()
                conn.commit()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return [{"lastrowid": cursor.lastrowid, "changes": cursor.rowcount}]
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
