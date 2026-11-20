"""Конфигурация бота. Значения берутся из переменных окружения и `bot/.env`.

Вложенные поля задаются через двойное подчёркивание: `LLM__BASE_URL` → `settings.llm.base_url`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BOT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BOT_DIR.parent


class LLMSettings(BaseModel):
    """OpenAI-совместимый сервер.

    - Yandex AI Studio: base_url `https://ai.api.cloud.yandex.net/v1`, `project` = id каталога,
      `model` = имя модели (`qwen3.6-35b-a3b`) — клиент сам соберёт URI `gpt://<каталог>/<модель>`;
    - Ollama: `http://host:11434/v1`; vLLM: `http://host:8000/v1`; `project` не нужен.
    """

    base_url: str = "https://ai.api.cloud.yandex.net/v1"
    model: str = "qwen3.6-35b-a3b"
    api_key: SecretStr = SecretStr("not-needed")
    project: str | None = None             # Yandex: id каталога (заголовок OpenAI-Project)
    timeout_s: float = 20.0
    max_retries: int = 1
    temperature: float = 0.1
    # seed поддерживается не всеми провайдерами (в документации Yandex не упомянут): None — не передаём
    seed: int | None = None
    max_tokens: int = 700
    # Режим рассуждений (Qwen3.x и др.). С ним рассуждение съедает max_tokens и ответ приходит пустым
    # (finish_reason=length), задержка растёт в разы. "none" — выключить; пусто — не передавать.
    reasoning_effort: str | None = "none"
    # json_schema → json_object → none: что поддерживает провайдер, проверяет polybot.check_llm
    response_format: Literal["json_schema", "json_object", "none"] = "json_schema"
    # Yandex: x-data-logging-enabled: false — запросы не используются для улучшения сервиса (C08)
    disable_data_logging: bool = True
    # Цена за 1000 токенов, ₽ (для стоимости прогона). Qwen3.6 35B в Yandex AI Studio: 0,2 / 0,3.
    price_in_per_1k: float = 0.2
    price_out_per_1k: float = 0.3

    @property
    def model_uri(self) -> str:
        if self.project and not self.model.startswith("gpt://"):
            return f"gpt://{self.project}/{self.model}"
        return self.model


class RetrievalSettings(BaseModel):
    # Qdrant: сервер (docker compose) или локальная папка (без Docker). Если задан qdrant_path,
    # используется он.
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_path: Path | None = BOT_DIR / "data" / "qdrant_local"
    collection: str = "polygraphy_kb"
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    models_dir: Path = BOT_DIR / "models"           # кэш весов fastembed (в .gitignore)
    timeout_s: float = 3.0
    max_retries: int = 1
    top_k: int = 5
    prefetch_k: int = 30                            # кандидатов от каждого из dense и BM25 до RRF
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    bm25_weight: float = 2.0                        # вес BM25 в RRF (dense = 1); выбран на dev 18.11
    # вес BM25 по заголовкам разделов (индекс schema 2); 0 — сигнал выключен
    title_weight: float = 1.0
    # Сколько чанков архива OmniDesk допускается в top_k (остальное — FAQ и руководство).
    # omnidesk_k ≥ top_k — без квоты, одна общая выдача.
    omnidesk_k: int = 2
    # evidence_gate: опора есть, если косинус лучшего чанка ≥ min_score; в промпт идут чанки
    # с косинусом ≥ min_chunk_score. Подбираются на dev (harness.retrieval_eval).
    min_score: float = 0.5
    min_chunk_score: float = 0.3
    # Индексация
    chunk_chars: int = 900
    chunk_overlap: int = 150

    @field_validator("qdrant_path", "models_dir", mode="before")
    @classmethod
    def _relative_to_bot(cls, v):
        """Пустая строка в .env — «не задано»; относительный путь — от папки bot/."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        p = Path(v)
        return p if p.is_absolute() else BOT_DIR / p


class LimitSettings(BaseModel):
    max_input_chars: int = 2000          # C01
    max_clarifications: int = 5          # C04, 2.3: на одну проблему
    max_steps: int = 10                  # бюджет шагов графа (самый длинный путь — 7 узлов)
    history_turns: int = 4               # сколько последних реплик идёт в промпт
    history_chars_per_turn: int = 500


class TelegramSettings(BaseModel):
    token: SecretStr = SecretStr("")      # от @BotFather; только в .env
    # Кто может писать боту, пока он в разработке: id пользователей Telegram через запятую.
    # Пусто — все. Узнать свой id: написать боту /id.
    allowed_user_ids: str = ""

    @property
    def allowed(self) -> set[int]:
        return {int(x) for x in self.allowed_user_ids.replace(" ", "").split(",") if x}


class StorageSettings(BaseModel):
    # memory — в памяти (тесты, харнесс); sqlite — переживает перезапуск (Telegram)
    backend: Literal["memory", "sqlite"] = "sqlite"
    sqlite_path: Path = BOT_DIR / "data" / "polybot.sqlite3"
    keep_history: int = 50               # сколько реплик чата хранить

    @field_validator("sqlite_path", mode="after")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return v if v.is_absolute() else BOT_DIR / v


class HandoffSettings(BaseModel):
    # Где хранится черновик, решает STORAGE__BACKEND: sqlite — таблица handoff_requests в той же
    # базе, что сессии (Telegram); харнесс и тесты держат черновики в памяти.
    timeout_s: float = 3.0


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
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    handoff: HandoffSettings = Field(default_factory=HandoffSettings)
    prompt_file: Path = BOT_DIR / "prompts" / "generate_v0.md"
    domain_classifier_dir: Path = REPO_DIR / "domain_classifier"
