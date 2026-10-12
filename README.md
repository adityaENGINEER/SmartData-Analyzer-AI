# 🩺 DataDoctor AI — Intelligent Data Quality & Analytics Agent

> **Diagnose Your Data. Fix the Issues. Discover the Insights.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 Problem Statement

Real-world datasets are rarely analysis-ready. Data analysts and scientists spend up to **80% of their time** diagnosing nulls, cleaning duplicate records, handling formatting anomalies, and troubleshooting mixed types before extracting any meaningful insight.

**DataDoctor AI** is a production-grade, interactive data quality and analytics agent designed to bridge this gap. It provides:
1. **Automated Data Profiling** — Immediate detection of missingness, duplicates, constant columns, whitespace inconsistencies, and statistical outliers.
2. **Explainable Health Score (0–100)** — Deterministic grading with transparent deduction breakdown.
3. **Smart, Non-Destructive Cleaning** — Interactive previews, approval gates, before-and-after audit logs, and an Undo stack.
4. **Natural Language Analytics & SQL** — Ask analytical questions locally or query via in-memory SQLite sandbox.
5. **Interactive Plotly Visualizations & Anomaly Detection** — IQR, Z-Score, and Scikit-Learn Isolation Forest.
6. **Executive Export Bundles** — Cleaned CSV/Excel datasets and markdown diagnostic health reports.

---

## 🏗️ Architecture Overview

```
DataDoctor-AI/
│
├── app.py                      # Application entrypoint & Overview Dashboard
├── requirements.txt            # Pinned stable package dependencies
├── .env.example                # Template for optional LLM provider keys
├── .gitignore                  # Git exclusions for secrets, temp files, and caches
│
├── pages/                      # Multi-Page Streamlit Dashboards
│   ├── 2_Data_Quality.py       # Granular diagnostics & explainable score
│   ├── 3_Smart_Cleaning.py     # Safe cleaning workspace with preview & undo
│   ├── 4_AI_Data_Analyst.py    # Natural language queries & SQLite playground
│   ├── 5_Visualizations.py     # Plotly interactive charts & recommendations
│   ├── 6_Anomaly_Detection.py  # IQR, Z-Score & Isolation Forest ML
│   ├── 7_Reports.py            # Cleaned data downloads & Markdown reports
│   └── 8_Settings.py           # Privacy guardrails & AI configuration
│
├── src/                        # Modular Core Business Logic
│   ├── data_loader.py          # Secure CSV & multi-sheet Excel ingestion
│   ├── data_profiler.py        # Structural & statistical anomaly profiling
│   ├── quality_score.py        # Explainable 0-100 quality scoring engine
│   ├── cleaning_engine.py      # Non-destructive transformation engine
│   ├── analytics_engine.py     # Deterministic local NLP computation engine
│   ├── visualization_engine.py # Plotly visualization builder & recommender
│   ├── anomaly_detector.py     # Univariate & multivariate anomaly detection
│   ├── sql_assistant.py        # In-memory SQLite executor with read-only guards
│   ├── report_generator.py     # Executive reports & export bytes generator
│   ├── ai_provider.py          # Optional LLM integration (OpenAI / Ollama)
│   ├── security.py             # File limits & SQL safety sanitization
│   └── ui_components.py        # Shared SaaS styling, undo stack & session state
│
├── tests/                      # Automated Test Suite (16 pytest tests)
│   ├── test_core.py            # Core engine unit tests
│   └── test_edge_cases.py      # Edge case & security validation tests
│
├── sample_data/                # Sample dirty datasets for demonstration
└── assets/                     # Architectural assets and documentation
```

---

## 🚀 Key Features

| Module | Features & Capabilities |
| :--- | :--- |
| **Data Ingestion** | Drag-and-drop CSV and multi-sheet Excel parsing with encoding fallback (`utf-8`, `latin-1`, `cp1252`) and 100 MB safety limits. |
| **Quality Engine** | Profiles 13+ structural checks: missing cells, duplicate rows, whitespace, constant features, date-as-text, and outlier heuristics. |
| **Explainable Score** | Evaluates **Completeness (35%)**, **Uniqueness (25%)**, **Consistency (20%)**, and **Integrity (20%)** with clear deduction reasons. |
| **Smart Cleaning** | Mean/Median/Mode imputation, de-duplication, text trimming, case normalization, and IQR/Z-score winsorization capping. |
| **Undo & Audit Trail** | Revert transformations instantly with one-click Undo; preserves the pristine original dataset in memory. |
| **AI Data Analyst** | Deterministic local Pandas analytical answers for averages, maximums, minimums, correlations, and summaries — 100% offline. Optional LLM fallback. |
| **SQL Playground** | Fast in-memory SQLite sandbox with query generation and strict read-only execution security (blocks destructive mutations). |
| **Visualizations** | Smart chart recommendations: Bar, Line, Scatter with OLS trendlines, Histograms, Box plots, and Pearson Correlation Heatmaps. |
| **Anomaly Studio** | Interquartile Range (IQR), Z-Score, and Scikit-Learn Isolation Forest with interactive 2D anomaly scatter plots. |
| **Reports & Export** | Export cleaned datasets as CSV or Excel (.xlsx) and download structured Markdown audit reports. |

---

## 💻 Installation & Setup (Windows & Cross-Platform)

### 1. Prerequisites
- **Python 3.10+** installed on your system.

### 2. Clone the Repository
```powershell
git clone https://github.com/your-username/DataDoctor-AI.git
cd DataDoctor-AI
```

### 3. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(On Linux/macOS: `source .venv/bin/activate`)*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 Running the Application

Launch the Streamlit web application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the complete 16-test automated suite using `pytest`:
```powershell
python -m pytest tests/ -v
```

---

## 🔒 Privacy & Security Guardrails

DataDoctor AI is designed with **privacy-by-default**:
- **Local Execution:** Uploaded files and Pandas dataframes are processed exclusively in your machine's volatile memory.
- **Zero Cloud Transmission:** Core profiling, cleaning, SQL queries, and anomaly detection transmit **zero bytes** to any cloud server.
- **Schema-Only Privacy for LLMs:** When optional AI features are invoked, only high-level column names and data types are transmitted. Bulk dataset rows and customer PII are strictly withheld.
- **Read-Only SQL Safety:** Destructive SQL statements (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, etc.) are proactively blocked.

---

## 🤖 Optional AI Configuration

DataDoctor AI runs **100% offline** without any API keys. If you want free-form conversational answers with an LLM:
1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Add your OpenAI API key or a local endpoint (Ollama / LM Studio):
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini
   ```
3. Alternatively, enter credentials directly in the **Settings** page within the app.

---

## 📸 Screenshots

| Overview & KPI Dashboard | Data Quality & Explainable Score |
| :---: | :---: |
| *(Hero KPI cards, dataset preview, schema breakdown)* | *(Health score breakdown, deductions, missingness chart)* |

| Smart Cleaning Workspace | Interactive Plotly Visualizations |
| :---: | :---: |
| *(Missing imputation, de-duplication, text trimming, undo)* | *(Correlation heatmaps, distribution histograms, scatter plots)* |

---

## 📋 Example Workflow

1. **Ingest:** Upload `sample_data/sample_sales_dirty.csv` or click **Load Built-in Sample Dataset** on the Overview page.
2. **Diagnose:** Review your **Data Quality Score** and detected issues on the **Data Quality** page.
3. **Clean:** Navigate to **Smart Cleaning** to impute missing sales, trim text whitespace, and deduplicate records.
4. **Explore:** Open **Visualizations** to inspect the Pearson correlation heatmap.
5. **Detect:** Run the **Isolation Forest** model in **Anomaly Detection** to isolate unusual transaction amounts.
6. **Query:** Ask analytical questions in the **AI Data Analyst** tab or execute SQLite queries.
7. **Export:** Download the cleaned dataset as CSV/Excel and export the full Markdown diagnostic report on the **Reports** page.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
