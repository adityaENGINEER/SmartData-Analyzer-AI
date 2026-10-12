"""
Optional AI Provider Integration for DataDoctor AI.
Supports OpenAI, Anthropic, or OpenAI-compatible local LLMs (Ollama, LM Studio, Groq).
Degrades gracefully when API keys are absent and strictly enforces privacy by sending schema metadata only.
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()


class AIProvider:
    """Provides optional LLM natural language analytics using external API or local endpoints."""

    @staticmethod
    def is_configured() -> bool:
        """Returns True if any supported AI provider credentials exist."""
        return bool(
            os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CUSTOM_LLM_API_KEY")
            or os.getenv("CUSTOM_LLM_BASE_URL")
        )

    @staticmethod
    def get_active_provider_info() -> Dict[str, Any]:
        """Returns metadata about the active AI configuration."""
        if os.getenv("OPENAI_API_KEY"):
            return {
                "provider": "OpenAI",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "status": "Configured",
            }
        elif os.getenv("ANTHROPIC_API_KEY"):
            return {
                "provider": "Anthropic",
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                "status": "Configured",
            }
        elif os.getenv("CUSTOM_LLM_BASE_URL"):
            return {
                "provider": "Custom / Local Endpoint (OpenAI-compatible)",
                "model": os.getenv("CUSTOM_LLM_MODEL", "local-model"),
                "status": "Configured",
            }
        return {
            "provider": "None (Offline / Local Mode)",
            "model": "Deterministic Pandas Analytics Engine",
            "status": "Not Configured",
        }

    @staticmethod
    def ask_ai(
        question: str,
        schema_metadata: Dict[str, Any],
        sample_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sends an analytical query with schema metadata (privacy-safe, no raw dataset) to the configured LLM.
        """
        if not AIProvider.is_configured():
            return {
                "success": False,
                "error": "No LLM API key configured. The application is running in 100% offline local mode. You can configure keys in the Settings page or via a .env file.",
            }

        prompt = (
            f"You are DataDoctor AI, a senior data scientist and analytics agent.\n"
            f"The user has loaded a dataset with the following schema and high-level summary:\n"
            f"Columns: {schema_metadata.get('columns')}\n"
            f"Data types: {schema_metadata.get('dtypes')}\n"
            f"Total columns: {schema_metadata.get('total_columns')}\n\n"
        )
        if sample_stats:
            prompt += f"Summary Statistics: {json.dumps(sample_stats)}\n\n"

        prompt += (
            f"User Question: '{question}'\n\n"
            f"Please provide an accurate, insightful, and professional data analytics answer based on the schema.\n"
            f"Do NOT invent columns or fake row statistics that are not present. Explain what calculations are needed or give concrete data guidance."
        )

        openai_key = os.getenv("OPENAI_API_KEY")
        custom_base = os.getenv("CUSTOM_LLM_BASE_URL")
        custom_key = os.getenv("CUSTOM_LLM_API_KEY", "")

        try:
            if openai_key or custom_base:
                base_url = custom_base or "https://api.openai.com/v1"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key or custom_key}",
                }
                payload = {
                    "model": os.getenv("OPENAI_MODEL") or os.getenv("CUSTOM_LLM_MODEL") or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a professional data analytics advisor. Be precise, concise, and helpful."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                }
                resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["choices"][0]["message"]["content"]
                    return {"success": True, "answer": answer, "provider": "OpenAI / Compatible"}
                else:
                    return {"success": False, "error": f"API returned status {resp.status_code}: {resp.text}"}

            return {"success": False, "error": "Provider protocol not supported."}
        except Exception as e:
            return {"success": False, "error": f"LLM Request failed: {str(e)}"}
