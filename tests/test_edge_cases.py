import os
import sys
import io
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.data_profiler import DataProfiler
from src.quality_score import QualityScoreCalculator
from src.cleaning_engine import CleaningEngine
from src.analytics_engine import AnalyticsEngine
from src.anomaly_detector import AnomalyDetector
from src.sql_assistant import SQLAssistant
from src.security import validate_sql_query, validate_file_metadata


def test_excel_dataset_loading():
    df_src = pd.DataFrame({
        "Product": ["Widget A", "Widget B", "Widget C"],
        "Units": [10, 25, 15],
        "Price": [19.99, 49.99, 29.99],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_src.to_excel(writer, index=False, sheet_name="Inventory")
    excel_bytes = buffer.getvalue()

    df_loaded, meta = DataLoader.load_dataset(excel_bytes, "test_inventory.xlsx", sheet_name="Inventory")
    assert len(df_loaded) == 3
    assert "Price" in df_loaded.columns
    assert meta["sheet_name"] == "Inventory"


def test_constant_column_detection():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Country": ["USA", "USA", "USA", "USA"],  # Constant
        "Score": [88, 92, 79, 85],
    })
    profile = DataProfiler.profile(df)
    assert "Country" in profile["constant_cols"]
    score_res = QualityScoreCalculator.calculate_score(profile)
    assert any(d["category"] == "consistency" for d in score_res["deductions"])


def test_extreme_missingness_and_empty_rows():
    df = pd.DataFrame({
        "A": [1, np.nan, 3, np.nan],
        "B": [np.nan, np.nan, np.nan, np.nan],  # Empty col
        "C": [10, np.nan, 30, np.nan],
    })
    profile = DataProfiler.profile(df)
    issue_types = [i["issue_type"] for i in profile["issues"]]
    assert "Empty Column" in issue_types
    assert "Empty Rows" in issue_types


def test_text_cleaning_operations():
    df = pd.DataFrame({"City": ["  new york  ", "LONDON", "  paris "]})
    clean_df, audit = CleaningEngine.apply_text_clean(df, "City", strip_whitespace=True, case_normalization="title")
    assert clean_df["City"].iloc[0] == "New York"
    assert clean_df["City"].iloc[1] == "London"
    assert clean_df["City"].iloc[2] == "Paris"


def test_outlier_capping():
    df = pd.DataFrame({"Score": [50, 52, 49, 51, 50, 53, 500]})  # 500 is outlier
    clean_df, audit = CleaningEngine.apply_outlier_handling(df, "Score", method="iqr", action="cap")
    assert clean_df["Score"].max() < 500
    assert len(clean_df) == 7  # All rows preserved during capping


def test_isolation_forest_minimum_rows():
    # Less than 20 rows should fail safely with informative explanation
    df = pd.DataFrame({"val1": [1, 2, 3], "val2": [4, 5, 6]})
    with pytest.raises(ValueError, match="at least 20 complete rows"):
        AnomalyDetector.detect_isolation_forest(df, ["val1", "val2"])


def test_unsupported_analytical_question():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    ans = AnalyticsEngine.answer_query(df, "What is the capital of France?")
    assert ans["matched"] is False
    assert "Supported local queries include" in ans["explanation"]
