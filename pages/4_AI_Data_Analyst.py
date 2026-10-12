"""
AI Data Analyst & SQL Assistant Page for DataDoctor AI.
Combines deterministic offline analytics, optional LLM natural language assistance, and in-memory SQLite querying.
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar
from src.analytics_engine import AnalyticsEngine
from src.sql_assistant import SQLAssistant
from src.ai_provider import AIProvider
from src.security import sanitize_schema_for_llm

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>🤖 AI Data Analyst & SQL Assistant</h1>
        <p>Ask natural-language analytical questions or explore data with safe read-only SQL.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

tab_qa, tab_sql = st.tabs(["💬 Natural Language Analytics", "⚡ In-Memory SQL Playground"])

# ----------------- 1. Natural Language Analyst -----------------
with tab_qa:
    st.markdown("#### Ask Analytical Questions")
    st.caption("Ask questions about averages, maximums, minimums, totals, missing values, or correlations.")

    provider_info = AIProvider.get_active_provider_info()
    st.markdown(
        f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <b>Engine:</b> {provider_info['provider']} ({provider_info['model']})
            </div>
            <div>
                <span class="badge {'badge-success' if provider_info['status'] == 'Configured' else 'badge-info'}">
                    ● {provider_info['status']}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Example question chips
    st.markdown("##### Quick Example Prompts:")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_q = ""
    with col_p1:
        if st.button("📈 Which columns correlate most strongly?", use_container_width=True):
            preset_q = "Which columns have the strongest correlation?"
    with col_p2:
        if st.button("📉 Which columns contain missing values?", use_container_width=True):
            preset_q = "Which columns contain the most missing values?"
    with col_p3:
        if st.button("📊 Summarize dataset patterns", use_container_width=True):
            preset_q = "Summarize the main patterns in this dataset."

    user_query = st.text_input("Enter your analytical question:", value=preset_q, placeholder="e.g. What is the average Salary? Or Which product has the highest revenue?")

    if st.button("Analyze Question", type="primary", key="btn_run_nlp") or (preset_q and user_query):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing question and computing statistics..."):
                # Always attempt deterministic local computation first for speed and privacy
                local_result = AnalyticsEngine.answer_query(df, user_query)

                if local_result["matched"]:
                    st.markdown("##### 💡 Analytical Result:")
                    st.success(local_result["answer"])
                    st.markdown(f"**Calculation Methodology:** {local_result['explanation']}")

                    if local_result["data"] is not None:
                        st.dataframe(local_result["data"], use_container_width=True)

                else:
                    # If local pattern didn't match and an LLM is configured, query the LLM safely
                    if AIProvider.is_configured():
                        st.info("Query requires advanced reasoning. Consulting configured LLM provider...")
                        schema_meta = sanitize_schema_for_llm(list(df.columns), df.dtypes.to_dict())
                        summary_stats = AnalyticsEngine.get_summary_statistics(df)
                        ai_resp = AIProvider.ask_ai(user_query, schema_meta, sample_stats=summary_stats)
                        if ai_resp["success"]:
                            st.markdown("##### 🤖 AI Analyst Assessment:")
                            st.markdown(ai_resp["answer"])
                        else:
                            st.error(f"AI Provider error: {ai_resp.get('error')}")
                    else:
                        st.warning("Local engine explanation:")
                        st.write(local_result["answer"])
                        st.info(local_result["explanation"])
                        st.caption("Tip: You can configure an OpenAI or Anthropic API key in Settings or .env to answer free-form natural language queries.")

# ----------------- 2. SQL Assistant -----------------
with tab_sql:
    st.markdown("#### Local SQLite Analytical Workspace")
    st.caption("Execute read-only SQL queries directly against your active DataFrame in local memory.")

    sql_assistant = SQLAssistant(df, table_name="dataset")

    # Schema view
    with st.expander("🔍 View Table Schema & Column Mapping", expanded=False):
        schema_info = sql_assistant.get_schema()
        st.dataframe(pd.DataFrame(schema_info), use_container_width=True)

    # Starter queries dropdown
    starter_queries = sql_assistant.generate_starter_queries()
    chosen_starter = st.selectbox("Select a starter query template:", ["None"] + [q["title"] for q in starter_queries])

    initial_sql = "SELECT * FROM dataset LIMIT 10;"
    if chosen_starter != "None":
        for q in starter_queries:
            if q["title"] == chosen_starter:
                initial_sql = q["sql"]
                break

    query_input = st.text_area("SQL Query (SELECT statements only):", value=initial_sql, height=120)

    if st.button("Run SQL Query", type="primary", key="btn_run_sql"):
        with st.spinner("Executing query in local SQLite engine..."):
            res_df, err = sql_assistant.execute_query(query_input)
            if err:
                st.error(err)
            else:
                st.success(f"Query returned {len(res_df):,} record(s).")
                st.dataframe(res_df, use_container_width=True)
