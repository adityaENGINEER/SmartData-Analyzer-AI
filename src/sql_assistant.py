"""
In-Memory SQLite Assistant for DataDoctor AI.
Allows executing secure read-only SQL queries on the active DataFrame and provides schema reflection.
"""

import sqlite3
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from src.security import validate_sql_query


class SQLAssistant:
    """Provides SQL schema exploration and safe in-memory analytical query execution."""

    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.table_name = table_name
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.column_names = list(df.columns)
        # Clean column names for SQL safety (replace spaces and special characters with underscores)
        self.safe_columns = {col: self._sanitize_col(col) for col in df.columns}

        # Create copy with safe names for SQL table
        renamed_df = df.copy()
        renamed_df.columns = [self.safe_columns[c] for c in df.columns]
        renamed_df.to_sql(self.table_name, self.conn, index=False, if_exists="replace")

    @staticmethod
    def _sanitize_col(name: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in str(name))
        if safe and safe[0].isdigit():
            safe = "_" + safe
        return safe or "col"

    def get_schema(self) -> List[Dict[str, str]]:
        """Returns SQLite table schema information."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name});")
        info = cursor.fetchall()
        # info columns: cid, name, type, notnull, dflt_value, pk
        return [{"cid": str(row[0]), "name": str(row[1]), "type": str(row[2])} for row in info]

    def execute_query(self, query: str, limit: int = 500) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Executes a validated read-only SQL query against the in-memory SQLite table.
        Returns (result_df, error_message).
        """
        # Validate read-only security
        is_valid, msg = validate_sql_query(query)
        if not is_valid:
            return None, f"Security Validation Failed: {msg}"

        try:
            # Append limit if not present and not an aggregation
            q_clean = query.strip().rstrip(";")
            result_df = pd.read_sql_query(q_clean, self.conn)
            if len(result_df) > limit:
                result_df = result_df.head(limit)
            return result_df, None
        except Exception as e:
            return None, f"SQL Execution Error: {str(e)}"

    def generate_starter_queries(self) -> List[Dict[str, str]]:
        """Generates helpful starter queries tailored to the dataset columns."""
        safe_cols = list(self.safe_columns.values())
        if not safe_cols:
            return []

        queries = [
            {
                "title": "Preview First 10 Records",
                "description": "Selects the first 10 rows from the dataset.",
                "sql": f"SELECT * FROM {self.table_name} LIMIT 10;",
            },
            {
                "title": "Row Count",
                "description": "Counts total number of rows in the table.",
                "sql": f"SELECT COUNT(*) AS total_records FROM {self.table_name};",
            },
        ]

        # Top category query if columns exist
        if len(safe_cols) > 0:
            first_col = safe_cols[0]
            queries.append({
                "title": f"Frequency Breakdown of {first_col}",
                "description": f"Aggregates record count grouped by {first_col}.",
                "sql": f"SELECT {first_col}, COUNT(*) AS frequency\nFROM {self.table_name}\nGROUP BY {first_col}\nORDER BY frequency DESC\nLIMIT 10;",
            })

        if len(safe_cols) > 1:
            queries.append({
                "title": f"Distinct Values Count",
                "description": "Calculates distinct values for selected features.",
                "sql": f"SELECT COUNT(DISTINCT {safe_cols[0]}) AS unique_{safe_cols[0]} FROM {self.table_name};",
            })

        return queries

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
