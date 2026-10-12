"""
Smart Data Cleaning Workspace for DataDoctor AI.
Provides previews, approval gates, before-and-after audit logs, and non-destructive transformations.
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar, refresh_profiling
from src.cleaning_engine import CleaningEngine

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>✨ Smart Data Cleaning Workspace</h1>
        <p>Safely transform data with real-time preview, before-and-after comparison, and audit trail.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

# Helper function to push state to undo stack before applying
def apply_and_log(new_df: pd.DataFrame, audit: dict):
    # Save current copy for Undo
    st.session_state.undo_stack.append((st.session_state.cleaned_df.copy(), audit.get("operation", "Cleaning Operation")))
    st.session_state.cleaned_df = new_df
    st.session_state.cleaning_history.append(audit)
    refresh_profiling()
    st.success(f"✅ Successfully applied: {audit.get('summary')}")
    st.rerun()

# ----------------- ⚡ 1-CLICK AUTO CLEAN BANNER -----------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); padding: 1.25rem 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; margin-bottom: 0.2rem;">⚡ Quick Auto-Clean (1-Click)</div>
            <div style="font-size: 0.85rem; color: #e0f2fe;">Automatically deduplicates, trims all text whitespace, and imputes missing values safely.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_auto1, col_auto2 = st.columns([1.5, 3])
with col_auto1:
    if st.button("🚀 Apply 1-Click Full Clean", type="primary", use_container_width=True):
        with st.spinner("Executing automated end-to-end cleaning..."):
            working_copy = df.copy()
            initial_rows = len(working_copy)
            operations_performed = []

            # 1. Deduplicate
            dups = int(working_copy.duplicated().sum())
            if dups > 0:
                working_copy, a_dup = CleaningEngine.apply_remove_duplicates(working_copy)
                operations_performed.append(f"Removed {dups} duplicate row(s)")

            # 2. Text clean all text columns
            text_cols = [c for c in working_copy.columns if not pd.api.types.is_numeric_dtype(working_copy[c]) and not pd.api.types.is_datetime64_any_dtype(working_copy[c])]
            trimmed_count = 0
            for tc in text_cols:
                working_copy[tc] = working_copy[tc].astype(str).str.strip().replace(r"\s+", " ", regex=True)
                trimmed_count += 1
            if trimmed_count > 0:
                operations_performed.append(f"Trimmed whitespace across {trimmed_count} text column(s)")

            # 3. Impute missing values (median for numeric, mode for text)
            imputed_cols = 0
            for col in working_copy.columns:
                if working_copy[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(working_copy[col]):
                        med = working_copy[col].median()
                        working_copy[col] = working_copy[col].fillna(med)
                        imputed_cols += 1
                    else:
                        modes = working_copy[col].mode(dropna=True)
                        if not modes.empty:
                            working_copy[col] = working_copy[col].fillna(modes.iloc[0])
                            imputed_cols += 1
            if imputed_cols > 0:
                operations_performed.append(f"Imputed nulls across {imputed_cols} column(s)")

            summary_text = "; ".join(operations_performed) if operations_performed else "No pending issues found to auto-clean."

            apply_and_log(working_copy, {
                "operation": "1-Click Auto-Clean",
                "column": "Dataset-wide",
                "rows_affected": initial_rows - len(working_copy) if dups > 0 else len(working_copy),
                "summary": summary_text,
            })

with col_auto2:
    st.caption("Pressing this automatically executes standard data hygiene (de-duplication, text whitespace trimming, and median/mode null filling) in one go. You can always click Undo in the sidebar to revert!")

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)


clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5, clean_tab6 = st.tabs([
    "1. Impute Missing Values",
    "2. Remove Duplicates",
    "3. Convert Data Types",
    "4. Text Cleaning & Whitespace",
    "5. Outlier Capping",
    "6. Drop Columns",
])

# ----------------- 1. Missing Values -----------------
with clean_tab1:
    st.markdown("#### Impute or Drop Missing Values")
    cols_with_na = [c for c in df.columns if df[c].isnull().sum() > 0]

    if not cols_with_na:
        st.success("No missing values found in any column!")
    else:
        col_sel = st.selectbox("Select column with missing values:", cols_with_na, key="sel_na_col")
        missing_count = int(df[col_sel].isnull().sum())
        missing_pct = round((missing_count / len(df)) * 100, 1)
        is_num = pd.api.types.is_numeric_dtype(df[col_sel])

        st.info(f"Column **'{col_sel}'** has **{missing_count}** missing value(s) ({missing_pct}% of rows).")

        strategies = ["drop_rows", "mode", "constant"]
        if is_num:
            strategies = ["mean", "median"] + strategies

        strat = st.selectbox("Choose Imputation Strategy:", strategies, format_func=lambda s: {
            "mean": "Mean (Average value)",
            "median": "Median (50th percentile)",
            "mode": "Mode (Most frequent category)",
            "constant": "Custom Constant Value",
            "drop_rows": "Drop Entire Rows containing NaN",
        }.get(s, s))

        custom_val = None
        if strat == "constant":
            custom_val = st.text_input("Enter replacement value:", value="Unknown")

        if st.button("Preview Imputation Result", key="btn_prev_na"):
            try:
                if strat == "drop_rows":
                    st.write(f"Will drop **{missing_count}** rows. New dataset length: {len(df) - missing_count} rows.")
                else:
                    new_ser, audit = CleaningEngine.preview_impute_missing(df, col_sel, strat, custom_val)
                    st.write("##### Sample Comparison (First 10 rows):")
                    comp_df = pd.DataFrame({
                        f"Original ({col_sel})": df[col_sel].head(10),
                        f"Transformed ({col_sel})": new_ser.head(10),
                    })
                    st.dataframe(comp_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing: {e}")

        if st.button("Apply Missing Value Fix", type="primary", key="btn_apply_na"):
            try:
                new_df, audit = CleaningEngine.apply_impute_missing(df, col_sel, strat, custom_val)
                apply_and_log(new_df, audit)
            except Exception as e:
                st.error(f"Failed to apply imputation: {e}")

# ----------------- 2. Remove Duplicates -----------------
with clean_tab2:
    st.markdown("#### Deduplicate Dataset Records")
    total_dups = int(df.duplicated().sum())

    if total_dups == 0:
        st.success("No exact duplicate rows found across all columns!")
    else:
        st.warning(f"Found **{total_dups}** exact duplicate row(s) ({round(total_dups/len(df)*100, 2)}%).")
        st.markdown("##### Preview Duplicated Records:")
        dup_preview = df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:2]))
        st.dataframe(dup_preview.head(20), use_container_width=True)

    st.markdown("##### De-duplication Scope:")
    use_subset = st.checkbox("Check duplicates based on specific key columns only", value=False)
    subset_cols = None
    if use_subset:
        subset_cols = st.multiselect("Select key columns:", options=list(df.columns))

    keep_strategy = st.radio("Which record to retain?", ["first", "last"], horizontal=True)

    if st.button("Remove Duplicates", type="primary", key="btn_apply_dup"):
        try:
            new_df, audit = CleaningEngine.apply_remove_duplicates(df, subset_cols=subset_cols, keep=keep_strategy)
            apply_and_log(new_df, audit)
        except Exception as e:
            st.error(f"Failed to remove duplicates: {e}")

# ----------------- 3. Convert Data Types -----------------
with clean_tab3:
    st.markdown("#### Safe Column Type Conversion")
    conv_col = st.selectbox("Select column to convert:", df.columns, key="sel_conv_col")
    curr_dtype = str(df[conv_col].dtype)
    st.write(f"Current Data Type: `{curr_dtype}`")

    target_type = st.selectbox("Select Target Type:", ["numeric", "datetime", "category", "string"], key="sel_target_type")

    if st.button("Apply Type Conversion", type="primary", key="btn_apply_conv"):
        try:
            new_df, audit = CleaningEngine.apply_type_conversion(df, conv_col, target_type)
            apply_and_log(new_df, audit)
        except Exception as e:
            st.error(f"Conversion failed: {e}")

# ----------------- 4. Text Cleaning -----------------
with clean_tab4:
    st.markdown("#### Text Cleaning & Whitespace Normalization")
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not text_cols:
        st.info("No text columns found in dataset.")
    else:
        txt_col = st.selectbox("Select text column to clean:", text_cols, key="sel_txt_col")
        strip_ws = st.checkbox("Trim leading and trailing whitespace", value=True)
        norm_spaces = st.checkbox("Normalize multiple internal spaces into single space", value=True)
        casing = st.selectbox("Case Normalization:", ["No change", "lowercase", "UPPERCASE", "Title Case"])

        case_map = {"No change": None, "lowercase": "lower", "UPPERCASE": "upper", "Title Case": "title"}

        if st.button("Apply Text Cleaning", type="primary", key="btn_apply_txt"):
            try:
                new_df, audit = CleaningEngine.apply_text_clean(
                    df,
                    txt_col,
                    strip_whitespace=strip_ws,
                    case_normalization=case_map[casing],
                    normalize_spaces=norm_spaces,
                )
                apply_and_log(new_df, audit)
            except Exception as e:
                st.error(f"Text cleaning failed: {e}")

# ----------------- 5. Outlier Capping -----------------
with clean_tab5:
    st.markdown("#### Statistical Outlier Capping (Winsorization)")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.info("No numeric columns available for outlier handling.")
    else:
        out_col = st.selectbox("Select numeric column:", num_cols, key="sel_out_col")
        method = st.selectbox("Outlier Threshold Method:", ["iqr", "zscore"], format_func=lambda m: "1.5x IQR (Boxplot bounds)" if m == "iqr" else "3x Z-Score (Normal distribution)")
        action = st.selectbox("Action to take:", ["cap", "remove"], format_func=lambda a: "Cap outliers to threshold boundaries (Preserves all rows)" if a == "cap" else "Remove rows containing outliers")

        if st.button("Apply Outlier Remediation", type="primary", key="btn_apply_out"):
            try:
                new_df, audit = CleaningEngine.apply_outlier_handling(df, out_col, method=method, action=action)
                apply_and_log(new_df, audit)
            except Exception as e:
                st.error(f"Outlier handling failed: {e}")

# ----------------- 6. Drop Columns -----------------
with clean_tab6:
    st.markdown("#### Drop Redundant or Constant Columns")
    drop_candidates = st.multiselect("Select columns to drop permanently from working copy:", df.columns)

    if st.button("Drop Selected Columns", type="primary", disabled=len(drop_candidates) == 0, key="btn_drop_cols"):
        try:
            new_df, audit = CleaningEngine.apply_drop_columns(df, drop_candidates)
            apply_and_log(new_df, audit)
        except Exception as e:
            st.error(f"Drop columns failed: {e}")

st.markdown("---")

# 7. Audit Trail Section
st.markdown("### 📜 Session Cleaning Audit Trail")
history = st.session_state.cleaning_history
if not history:
    st.info("No transformations applied yet. Use the tabs above to clean your dataset.")
else:
    hist_df = pd.DataFrame([
        {
            "Step": idx,
            "Operation": h["operation"],
            "Column(s)": h["column"],
            "Rows Affected": h["rows_affected"],
            "Summary": h["summary"],
        }
        for idx, h in enumerate(history, 1)
    ])
    st.dataframe(hist_df, use_container_width=True)
