"""
Settings, Privacy & LLM Credentials Configuration Page for DataDoctor AI.
"""

import os
import streamlit as st
from src.ui_components import init_session_state, inject_custom_css, render_sidebar
from src.ai_provider import AIProvider

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>⚙️ System Settings & AI Configuration</h1>
        <p>Manage model credentials, inspect privacy policies, and configure data limits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_keys, tab_privacy, tab_limits = st.tabs(["🔑 AI Provider Keys", "🔒 Privacy & Security Guardrails", "⚙️ Processing Limits"])

# ----------------- 1. AI Keys -----------------
with tab_keys:
    st.markdown("#### Optional AI Model Configuration")
    st.markdown(
        """
        DataDoctor AI executes data profiling, cleaning, statistical anomaly detection, and local analytics 
        **100% locally with NO API key required**. 
        If you wish to enable free-form conversational reasoning with external LLMs, you can configure your keys below.
        """
    )

    provider_info = AIProvider.get_active_provider_info()
    st.info(f"**Current Active Provider:** {provider_info['provider']} ({provider_info['model']}) — Status: {provider_info['status']}")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        openai_key = st.text_input("OpenAI API Key:", value=os.getenv("OPENAI_API_KEY", ""), type="password", placeholder="sk-...")
        openai_model = st.selectbox("OpenAI Model:", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    with col_k2:
        custom_base = st.text_input("Local / Custom Endpoint URL (e.g. Ollama / LM Studio):", value=os.getenv("CUSTOM_LLM_BASE_URL", ""), placeholder="http://localhost:11434/v1")
        custom_key = st.text_input("Custom Endpoint API Key (Optional):", value=os.getenv("CUSTOM_LLM_API_KEY", ""), type="password")

    if st.button("Save Credentials to Current Session", type="primary"):
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["OPENAI_MODEL"] = openai_model
        if custom_base:
            os.environ["CUSTOM_LLM_BASE_URL"] = custom_base
            os.environ["CUSTOM_LLM_API_KEY"] = custom_key

        st.success("Session credentials updated successfully!")
        st.rerun()

# ----------------- 2. Privacy Guardrails -----------------
with tab_privacy:
    st.markdown("#### Privacy & Local Processing Architecture")
    st.markdown(
        """
        - **Local-First Execution:** Uploaded files and Pandas dataframes are processed exclusively in your machine's volatile memory.
        - **Zero Cloud Transmission by Default:** The core profiling, cleaning engine, SQL sandbox, and visualizations transmit zero data to third-party servers.
        - **Schema Redaction for LLMs:** When optional AI features are invoked, only high-level column names and data types are transmitted. Raw row contents and customer PII are strictly withheld.
        - **Read-Only SQL Safety:** Destructive SQL statements (DROP, DELETE, UPDATE, INSERT, ALTER) are automatically blocked before reaching the in-memory SQLite engine.
        """
    )

# ----------------- 3. Processing Limits -----------------
with tab_limits:
    st.markdown("#### Configured Safety Boundaries")
    st.write("- **Maximum Upload File Size:** 100 MB")
    st.write("- **Supported Extensions:** `.csv`, `.xlsx`, `.xls`")
    st.write("- **SQL Query Row Output Cap:** 500 records")
    st.write("- **Scatter Plot Trendline Cap:** 5,000 points (prevents OLS memory spikes)")
