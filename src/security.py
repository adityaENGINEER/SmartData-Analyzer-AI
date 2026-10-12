"""
Data Security and Privacy Guardrails for DataDoctor AI.
Enforces local-only processing by default, payload redaction, and safe SQL sanitization.
"""

import re
from typing import Tuple, List, Dict, Any

# Max file size: 100 MB default
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Disallowed SQL statement keywords for read-only analytical execution
DISALLOWED_SQL_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bTRUNCATE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bREINDEX\b",
    r"\bVACUUM\b",
    r"\bPRAGMA\b",
    r"\bEXEC\b",
    r"\bEXECUTE\b",
]

DISALLOWED_SQL_PATTERN = re.compile("|".join(DISALLOWED_SQL_KEYWORDS), re.IGNORECASE)


def validate_file_metadata(filename: str, file_size: int) -> Tuple[bool, str]:
    """Validate uploaded file extension and size safely."""
    if not filename:
        return False, "File name cannot be empty."

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if file_size <= 0:
        return False, "Uploaded file is empty (0 bytes)."

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, f"File exceeds maximum allowed size of {max_mb} MB."

    return True, "Valid file"


def validate_sql_query(query: str) -> Tuple[bool, str]:
    """
    Validate that an analytical SQL query is purely read-only (SELECT statements only).
    Blocks arbitrary destructive statements (DROP, DELETE, UPDATE, INSERT, ALTER, etc.).
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."

    cleaned_query = query.strip().rstrip(";")

    # Check for multiple statements separated by semicolon
    if ";" in cleaned_query:
        return False, "Execution of multiple SQL statements is disallowed for security."

    # Must start with SELECT or WITH (Common Table Expressions) or EXPLAIN
    first_word = cleaned_query.split()[0].upper()
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        return False, f"Only read-only analytical queries (SELECT or WITH) are permitted. Received: {first_word}"

    # Check for disallowed mutation keywords
    match = DISALLOWED_SQL_PATTERN.search(cleaned_query)
    if match:
        return False, f"Disallowed destructive keyword detected in query: '{match.group(0)}'"

    return True, "Valid analytical query"


def sanitize_schema_for_llm(df_columns: List[str], dtypes: Dict[str, Any], sample_rows: int = 3) -> Dict[str, Any]:
    """
    Prepares a privacy-preserving metadata schema for natural language assistance.
    Ensures that raw bulk customer PII is never transmitted by default.
    """
    return {
        "columns": df_columns,
        "dtypes": {str(k): str(v) for k, v in dtypes.items()},
        "total_columns": len(df_columns),
    }
