import os
import sys
import pytest
import pandas as pd
import numpy as np 

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.data_profiler import DataProfiler
from src.quality_score import QualityScoreCalculator
from src.cleaning_engine import CleaningEngine
from src.analytics_engine import AnalyticsEngine
from src.anomaly_detector import AnomalyDetector
from src.security import validate_sql_query, validate_file_metadata
from src.sql_assistant import SQLAssistant


@pytest.fixture
def sample_csv_data():
    csv_str = (
        "Name,Age,Salary,Department,JoinDate\n"
        "Alice,25,50000,Engineering,2022-01-15\n"
        "Bob,30,60000,Marketing,2021-03-10\n"
        "Charlie,,75000,Engineering,2020-07-22\n"
        "David,45,120000,Executive,2019-11-01\n"
        "Eve,28,52000,Marketing,2023-02-18\n"
        "Eve,28,52000,Marketing,2023-02-18\n"  # Duplicate row
        "Frank,  32  ,58000, Sales ,2022-09-01\n"  # Whitespace
    )
    return csv_str.encode("utf-8")


def test_data_loader_csv(sample_csv_data):
    df, meta = DataLoader.load_dataset(sample_csv_data, "test.csv")
    assert df is not None
    assert len(df) == 7
    assert meta["col_count"] == 5
    assert "Salary" in df.columns


def test_data_loader_empty_file():
    with pytest.raises(ValueError, match="empty"):
        DataLoader.load_dataset(b"", "empty.csv")


def test_data_profiler(sample_csv_data):
    df, _ = DataLoader.load_dataset(sample_csv_data, "test.csv")
    profile = DataProfiler.profile(df)
    assert profile["total_rows"] == 7
    assert profile["duplicate_rows"] == 1
    assert profile["missing_cells"] >= 1

    issue_types = [i["issue_type"] for i in profile["issues"]]
    assert "Duplicate Rows" in issue_types
    assert "Missing Values" in issue_types


def test_quality_score(sample_csv_data):
    df, _ = DataLoader.load_dataset(sample_csv_data, "test.csv")
    profile = DataProfiler.profile(df)
    score_res = QualityScoreCalculator.calculate_score(profile)
    assert 0 <= score_res["overall_score"] <= 100
    assert "grade" in score_res
    assert len(score_res["deductions"]) > 0


def test_cleaning_engine_preserves_original(sample_csv_data):
    df, _ = DataLoader.load_dataset(sample_csv_data, "test.csv")
    original_len = len(df)

    # Remove duplicates
    cleaned_df, audit = CleaningEngine.apply_remove_duplicates(df)
    assert len(cleaned_df) == original_len - 1
    # Verify original df is untouched
    assert len(df) == original_len

    # Impute missing Age with median
    # In df, Age has strings/whitespaces or missing
    cleaned_df, _ = CleaningEngine.apply_type_conversion(cleaned_df, "Age", "numeric")
    imputed_df, audit_imp = CleaningEngine.apply_impute_missing(cleaned_df, "Age", "median")
    assert imputed_df["Age"].isnull().sum() == 0
    assert cleaned_df["Age"].isnull().sum() > 0  # previous df untouched


def test_sql_security():
    assert validate_sql_query("SELECT * FROM dataset;")[0] is True
    assert validate_sql_query("WITH cte AS (SELECT 1) SELECT * FROM cte;")[0] is True

    # Disallowed destructive queries
    assert validate_sql_query("DROP TABLE dataset;")[0] is False
    assert validate_sql_query("DELETE FROM dataset WHERE id=1;")[0] is False
    assert validate_sql_query("INSERT INTO dataset VALUES (1);")[0] is False
    assert validate_sql_query("SELECT 1; DROP TABLE dataset;")[0] is False


def test_sql_assistant_execution():
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    assistant = SQLAssistant(df, "test_table")
    res, err = assistant.execute_query("SELECT SUM(val) AS total FROM test_table")
    assert err is None
    assert res is not None
    assert res["total"].iloc[0] == 60
    assistant.close()


def test_analytics_engine():
    df = pd.DataFrame({
        "Revenue": [100, 200, 300, 400],
        "Cost": [50, 100, 150, 200],
        "Category": ["A", "B", "A", "B"],
    })
    ans = AnalyticsEngine.answer_query(df, "What is the average Revenue?")
    assert ans["matched"] is True
    assert "250" in ans["answer"]

    ans_corr = AnalyticsEngine.answer_query(df, "Show correlation")
    assert ans_corr["matched"] is True
    assert "Revenue" in ans_corr["answer"]


def test_anomaly_detector():
    df = pd.DataFrame({"values": [10, 12, 11, 13, 12, 10, 11, 100]})  # 100 is outlier
    res = AnomalyDetector.detect_iqr(df, "values")
    assert res["anomaly_count"] == 1
    assert 7 in res["flagged_indices"]
