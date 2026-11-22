"""Telegram-бот: тонкая обёртка над `Bot.respond` — тот же путь, что у харнесса.

    .venv/bin/python -m polybot.telegram_bot

Нужен TELEGRAM__TOKEN в .env (от @BotFather). Пока бот в разработке, доступ можно
ограничить: TELEGRAM__ALLOWED_USER_IDS=123,456 (свой id бот пришлёт на /id).

Команды: /start — о боте, /new — новая проблема (сброс уточнений), /id — ваш id.
Под содержательным ответом — оценка «👍 Полезно» / «👎 Не помогло» (M09, таблица feedback) и
«Проблема решена»; под уточнением — только «Проблема решена» (то же, что /new). Под предложением
консультанта — «Сохранить черновик» и «Не нужно» (C06): без нажатия или явного «да» ничего
не передаётся. Черновики обращений — в той же SQLite, таблица handoff_requests.
Состояние чатов и журнал ходов — в SQLite (STORAGE__SQLITE_PATH), переживают перезапуск.

Локальный Qdrant открывается только одним процессом: пока бот запущен, харнесс и
check_llm с тем же индексом не запустятся (или поднимите Qdrant в Docker, RETRIEVAL__QDRANT_URL).
"""
from __future__ import annotations

import asyncio
import html
import logging
import re

from polybot import texts
from polybot.bot import Bot, SessionStore
from polybot.schemas import Response
from polybot.settings import Settings

logger = logging.getLogger("polybot.telegram")

TG_LIMIT = 4096
SOLVED = "solved"
SOLVED_LABEL = "Проблема решена"
REPEAT_PREFIX = "repeat:"
# действие из Response.actions → (callback_data, подпись кнопки)
BUTTONS = {"handoff_confirm": ("handoff_confirm", "Сохранить черновик"),
           "handoff_decline": ("handoff_decline", "Не нужно")}
# Оценка ответа (M09): callback_data «fb:up:<trace_id>» — оценка привязана к конкретному ходу.
FEEDBACK = {"feedback_up": ("up", "👍 Полезно"), "feedback_down": ("down", "👎 Не помогло")}
FEEDBACK_THANKS = "Спасибо за оценку"


def keyboard_rows(actions: list[str], trace_id: str,
                  offer_id: str | None = None) -> list[list[tuple[str, str]]]:
    """Кнопки ответа рядами (подпись, callback_data): оценка — отдельным рядом сверху."""
    fb = [(FEEDBACK[a][1], f"fb:{FEEDBACK[a][0]}:{trace_id}") for a in actions if a in FEEDBACK]
    rest = []
    for action in actions:
        if action == SOLVED:
            rest.append((SOLVED_LABEL, f"{SOLVED}:{trace_id}"))
        elif action in BUTTONS and offer_id:
            callback, label = BUTTONS[action]
            rest.append((label, f"{callback}:{offer_id}"))
    return [row for row in (fb, rest) if row]
# Первый экран — design doc 2.3 «Прозрачность»
START = (
    "Здравствуйте! Я отвечаю на вопросы по 1С:Полиграфии, КП/ИТС, лицензированию и поддержке "
    "по официальной документации и базе знаний, со ссылкой на источник. Я могу ошибаться — "
    "сверяйтесь с источником под ответом.\n\n"
    "Опишите вопрос одним сообщением: что делаете, в каком документе и что не получается. "
    "Если понадобится, я уточню детали.\n\n"
    "/new — начать новый вопрос."
)
NEW_PROBLEM = "Хорошо, начнём новый вопрос. Опишите, что нужно сделать."
NOT_TEXT = "Я пока понимаю только текстовые сообщения. Опишите, пожалуйста, вопрос текстом."
NOT_ALLOWED = "Бот сейчас в разработке, доступ ограничен."
OMNIDESK_SOURCE = "Архив ответов поддержки 1С:Полиграфии"



def session_id(chat_id: int) -> str:
    return f"tg:{chat_id}"


def format_sources(resp: Response) -> list[str]:
    lines, seen = [], set()
    for s in resp.sources:
        if s.doc_id.startswith("omnidesk:"):
            line = html.escape(OMNIDESK_SOURCE)           # обращения клиентов не цитируем
        elif s.url.startswith(("http://", "https://")):
            line = f'<a href="{html.escape(s.url, quote=True)}">{html.escape(s.title or s.url)}</a>'
        else:                                             # руководство: «Руководство …, ред. …»
            line = html.escape(" — ".join(x for x in (s.title, s.url) if x))
        if line not in seen:
            seen.add(line)
            lines.append(f"• {line}")
    return lines


BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.M)
BULLET_RE = re.compile(r"^(\s*)[*\-•]\s+", re.M)
CODE_RE = re.compile(r"`([^`\n]+)`")


def markdown_to_html(text: str) -> str:
    """Модель просят писать без Markdown, но если разметка всё же пришла, Telegram показал бы
    её звёздочками. Жирный и заголовки → <b>, маркеры списков → «–», `код` → текст,
    оставшиеся одиночные * и ** убираются. На входе — уже экранированный HTML."""
    text = HEADER_RE.sub(lambda m: f"**{m.group(1).strip()}**", text)
    text = BULLET_RE.sub(lambda m: f"{m.group(1)}– ", text)
    text = BOLD_RE.sub(lambda m: f"\x00{m.group(1) or m.group(2)}\x01", text)
    text = CODE_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("*", "")
    return text.replace("\x00", "<b>").replace("\x01", "</b>")


def format_reply(resp: Response) -> str:
    """Текст ответа в HTML Telegram: экранированный ответ + список источников."""
    body = markdown_to_html(html.escape(resp.text))
    src = format_sources(resp)
    if src:
        body += "\n\n<b>Источники:</b>\n" + "\n".join(src)
    return body


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Режем по абзацам/строкам, чтобы не порвать HTML-тег ссылки."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:                          # очень длинная строка без переносов
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        cand = f"{cur}\n{line}" if cur else line
        if len(cand) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts


def build_handoff(settings: Settings):
    if settings.storage.backend == "sqlite":
        from polybot.clients.handoff import SQLiteHandoff

        return SQLiteHandoff(settings.storage.sqlite_path)
    return None


def build_store(settings: Settings):
    if settings.storage.backend == "sqlite":
        from polybot.storage import SQLiteSessionStore

        return SQLiteSessionStore(settings.storage.sqlite_path, settings.storage.keep_history)
    return SessionStore()


def build_dispatcher(core: Bot, allowed: set[int]):
    from aiogram import Dispatcher, F, Router
    from aiogram.filters import Command, CommandStart
    from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
    from aiogram.utils.chat_action import ChatActionSender

    router = Router()
    router.message.filter(F.chat.type == "private")

    def markup(rows: list[list[tuple[str, str]]]):
        if not rows:
            return None
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=data) for label, data in row] for row in rows])

    async def send(m: Message, resp: Response) -> None:
        parts = split_message(format_reply(resp))
        rows = keyboard_rows(resp.actions, resp.trace_id, resp.handoff_offer_id)
        for i, part in enumerate(parts):
            await m.answer(part, parse_mode="HTML", disable_web_page_preview=True,
                           reply_markup=markup(rows) if i == len(parts) - 1 else None)

    def permitted(user) -> bool:
        return not allowed or (user is not None and user.id in allowed)

    @router.message(Command("id"))
    async def cmd_id(m: Message) -> None:
        await m.answer(f"Ваш id: {m.from_user.id}")

    @router.message(CommandStart())
    async def cmd_start(m: Message) -> None:
        if not permitted(m.from_user):
            await m.answer(NOT_ALLOWED)
            return
        await m.answer(START)

    @router.message(Command("new"))
    async def cmd_new(m: Message) -> None:
        if not permitted(m.from_user):
            await m.answer(NOT_ALLOWED)
            return
        core.new_problem(session_id(m.chat.id))
        await m.answer(NEW_PROBLEM)

    @router.callback_query((F.data == SOLVED) | F.data.startswith(f"{SOLVED}:"))
    async def cb_solved(q: CallbackQuery) -> None:
        if not permitted(q.from_user) or q.message is None:
            await q.answer()
            return
        _, _, trace_id = q.data.partition(":")
        if not trace_id or not await core.solve_problem(session_id(q.message.chat.id), trace_id):
            await q.answer("Эта кнопка относится к прошлому ответу. Используйте кнопку под текущим ответом.")
            return
        await q.answer("Отметила, спасибо")
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:                                 # сообщение старое или уже без кнопки
            pass
        repeat_rows = [[("Та же проблема снова", f"{REPEAT_PREFIX}{trace_id}")]]
        await q.message.answer(NEW_PROBLEM, reply_markup=markup(repeat_rows))

    @router.callback_query(F.data.startswith(REPEAT_PREFIX))
    async def cb_repeat(q: CallbackQuery) -> None:
        if not permitted(q.from_user) or q.message is None:
            await q.answer()
            return
        trace_id = q.data.removeprefix(REPEAT_PREFIX)
        if not await core.mark_repeated_problem(session_id(q.message.chat.id), trace_id):
            await q.answer("Эта отметка уже сохранена или срок отметки истёк.")
            return
        await q.answer("Отметила, что проблема повторилась.")
        await q.message.answer("Опишите, что происходит сейчас.")

    @router.callback_query(F.data.startswith("fb:"))
    async def cb_feedback(q: CallbackQuery) -> None:
        if not permitted(q.from_user) or q.message is None:
            await q.answer()
            return
        _, rating, trace_id = q.data.split(":", 2)
        sid = session_id(q.message.chat.id)
        core.feedback(sid, trace_id, rating)
        logger.info("%s feedback %s %s", sid, rating, trace_id)
        await q.answer(FEEDBACK_THANKS)
        # Ряд оценки убираем, «Проблема решена» оставляем; повторная оценка заменила бы первую.
        try:
            rest = [row for row in (q.message.reply_markup.inline_keyboard if q.message.reply_markup else [])
                    if not any((b.callback_data or "").startswith("fb:") for b in row)]
            await q.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=rest) if rest else None)
        except Exception:                                 # сообщение старое или уже без кнопок
            pass

    @router.callback_query(F.data.in_(set(BUTTONS)) | F.data.startswith("handoff_confirm:")
                           | F.data.startswith("handoff_decline:"))
    async def cb_handoff(q: CallbackQuery) -> None:
        if not permitted(q.from_user) or q.message is None:
            await q.answer()
            return
        action, _, offer_id = q.data.partition(":")
        sid = session_id(q.message.chat.id)
        if not offer_id:
            current = await core.refresh_handoff_offer(sid)
            if current is None:
                await q.answer(texts.HANDOFF_STALE)
            else:
                await q.answer("Подтвердите черновик новой кнопкой.")
                await send(q.message, current)
            return
        await q.answer()
        resp = await core.handoff_action(sid, action, offer_id=offer_id)
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        logger.info("%s %s %s %s", sid, action, resp.reason, resp.trace_id)
        await send(q.message, resp)

    @router.message(F.text)
    async def on_text(m: Message) -> None:
        if not permitted(m.from_user):
            await m.answer(NOT_ALLOWED)
            return
        sid = session_id(m.chat.id)
        try:
            async with ChatActionSender.typing(bot=m.bot, chat_id=m.chat.id):
                resp = await core.respond(sid, m.text)
        except Exception:
            logger.exception("ход упал вне графа: %s", sid)
            await m.answer(texts.FALLBACK_TOOL)
            return
        logger.info("%s %s %s %s мс %.3f ₽ %s", sid, resp.kind, resp.reason, resp.latency_ms,
                    resp.llm.cost_rub, resp.trace_id)
        await send(m, resp)

    @router.message()
    async def on_other(m: Message) -> None:
        await m.answer(NOT_TEXT)

    dp = Dispatcher()
    dp.include_router(router)
    return dp


async def main() -> None:
    from aiogram import Bot as TgBot
    from aiogram.types import BotCommand

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    token = settings.telegram.token.get_secret_value()
    if not token:
        raise SystemExit("нет TELEGRAM__TOKEN в bot/.env — создайте бота у @BotFather")
    store = build_store(settings)
    handoff = build_handoff(settings)
    core = Bot.from_settings(settings, store=store, handoff=handoff)
    tg = TgBot(token)
    await tg.set_my_commands([BotCommand(command="new", description="Новый вопрос"),
                              BotCommand(command="start", description="О боте")])
    allowed = settings.telegram.allowed
    logger.info("бот запущен; доступ: %s", sorted(allowed) if allowed else "все")
    try:
        await build_dispatcher(core, allowed).start_polling(tg)
    finally:
        await tg.session.close()
        close = getattr(core.services.retriever, "close", None)
        if close:
            await close()
        for res in (store, handoff):
            if hasattr(res, "close"):
                res.close()


if __name__ == "__main__":
    asyncio.run(main())
