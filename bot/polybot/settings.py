"""Конфигурация бота. Значения берутся из переменных окружения и `bot/.env`.

Вложенные поля задаются через двойное подчёркивание: `LLM__BASE_URL` → `settings.llm.base_url`.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BOT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BOT_DIR.parent


class LLMSettings(BaseModel):
    # OpenAI-совместимый сервер: Ollama (http://host:11434/v1) или vLLM (http://host:8000/v1).
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:4b"
    api_key: str = "not-needed"
    timeout_s: float = 20.0
    max_retries: int = 1
    temperature: float = 0.1
    seed: int = 42
    max_tokens: int = 700


class RetrievalSettings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    collection: str = "polygraphy_kb"
    timeout_s: float = 3.0
    top_k: int = 5
    # Порог evidence_gate: ниже — «опоры нет». Подбирается на dev на этапе 1.
    min_score: float = 0.5


class LimitSettings(BaseModel):
    max_input_chars: int = 2000          # C01
    max_clarifications: int = 5          # C04, 2.3: на одну проблему
    max_steps: int = 10                  # бюджет шагов графа (самый длинный путь — 7 узлов)
    history_turns: int = 4               # сколько последних реплик идёт в промпт
    history_chars_per_turn: int = 500


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    prompt_file: Path = BOT_DIR / "prompts" / "generate_v0.md"
    domain_classifier_dir: Path = REPO_DIR / "domain_classifier"
