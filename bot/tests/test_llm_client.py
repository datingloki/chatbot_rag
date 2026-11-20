"""Параметры запроса к OpenAI-совместимому провайдеру — без сетевых вызовов."""
import asyncio

from polybot.clients.llm import OpenAICompatibleLLM, status_kind
from polybot.errors import ToolErrorKind
from polybot.nodes.rag import REPLY_SCHEMA, strip_fences
from polybot.settings import LLMSettings
from tests.conftest import CHUNK, ask, make_bot, reply
from polybot.clients.llm import ScriptedLLM, LLMResult

MSG = [{"role": "user", "content": "привет"}]


def test_yandex_model_uri_built_from_folder():
    cfg = LLMSettings(project="b1gabc", model="qwen3.6-35b-a3b")
    assert cfg.model_uri == "gpt://b1gabc/qwen3.6-35b-a3b"
    assert LLMSettings(project="b1gabc", model="gpt://b1gabc/x").model_uri == "gpt://b1gabc/x"
    assert LLMSettings(project=None, model="qwen3:8b").model_uri == "qwen3:8b"


def test_request_params_and_headers():
    llm = OpenAICompatibleLLM(LLMSettings(api_key="k", project="f1", model="m"))
    params = llm.build_params(MSG, REPLY_SCHEMA)
    assert params["model"] == "gpt://f1/m"
    assert "seed" not in params                      # seed по умолчанию не передаём
    assert params["response_format"]["type"] == "json_schema"
    assert llm._client.default_headers["x-data-logging-enabled"] == "false"
    assert llm._client.project == "f1"


def test_response_format_modes():
    for mode, expected in (("json_object", {"type": "json_object"}), ("none", None)):
        llm = OpenAICompatibleLLM(LLMSettings(response_format=mode))
        assert llm.build_params(MSG, REPLY_SCHEMA).get("response_format") == expected


def test_reasoning_disabled_by_default():
    assert OpenAICompatibleLLM(LLMSettings()).build_params(MSG, None)["extra_body"] == {"reasoning_effort": "none"}
    assert "extra_body" not in OpenAICompatibleLLM(LLMSettings(reasoning_effort="")).build_params(MSG, None)


def test_seed_passed_when_set():
    llm = OpenAICompatibleLLM(LLMSettings(seed=7))
    assert llm.build_params(MSG, None)["seed"] == 7


def test_api_key_not_in_repr():
    assert "secret-key" not in repr(LLMSettings(api_key="secret-key"))


def test_status_codes_mapped():
    assert status_kind(401) is ToolErrorKind.AUTH
    assert status_kind(429) is ToolErrorKind.RATE_LIMITED
    assert status_kind(400) is ToolErrorKind.BAD_REQUEST
    assert status_kind(503) is ToolErrorKind.UNAVAILABLE


def test_json_in_code_fence_accepted():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'
    bot = make_bot(llm=ScriptedLLM("```json\n" + reply("answer", "Ответ.", [CHUNK.chunk_id]) + "\n```"))
    assert ask(bot, "вопрос").kind == "answer"


def test_cost_counted_from_tokens():
    class Counting(ScriptedLLM):
        async def complete(self, messages, json_schema=None):
            r = await super().complete(messages, json_schema)
            return LLMResult(r.text, "m", prompt_tokens=3000, completion_tokens=400, latency_ms=1.0)

    bot = make_bot(llm=Counting(reply("answer", "Ответ.", [CHUNK.chunk_id])))
    r = ask(bot, "вопрос")
    # 3000 × 0,2 / 1000 + 400 × 0,3 / 1000 = 0,72 ₽
    assert r.llm.cost_rub == 0.72
