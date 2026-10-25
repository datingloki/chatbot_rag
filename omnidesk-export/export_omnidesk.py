
import json
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


# --------------------------------------------------
# Настройки
# --------------------------------------------------

load_dotenv()

SUBDOMAIN = os.getenv("OMNIDESK_SUBDOMAIN")
EMAIL = os.getenv("OMNIDESK_EMAIL")
API_KEY = os.getenv("OMNIDESK_API_KEY")

if not SUBDOMAIN or not EMAIL or not API_KEY:
    raise SystemExit(
        "Не заданы OMNIDESK_SUBDOMAIN, OMNIDESK_EMAIL "
        "или OMNIDESK_API_KEY в файле .env"
    )

BASE_URL = f"https://{SUBDOMAIN}.omnidesk.ru/api"

OUTPUT_DIR = Path("output")
DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
ERRORS_PATH = OUTPUT_DIR / "errors.jsonl"
REPORT_PATH = OUTPUT_DIR / "report.json"

LIMIT = 100
REQUEST_DELAY = 0.5
MAX_SERVER_ERROR_RETRIES = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

session = requests.Session()
session.auth = HTTPBasicAuth(EMAIL, API_KEY)
session.headers.update({"Accept": "application/json"})


# --------------------------------------------------
# HTTP-запросы и обработка лимитов API
# --------------------------------------------------

def get_json(path, params=None):
    """Выполняет GET-запрос с обработкой лимитов и временных ошибок."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    server_error_retries = 0

    while True:
        try:
            response = session.get(
                url,
                params=params,
                timeout=(10, 60),
            )
        except requests.RequestException:
            logging.exception("Ошибка соединения при запросе %s", path)
            raise

        if response.status_code == 429:
            retry_after = response.headers.get("retry_after")

            try:
                wait_seconds = float(retry_after)
            except (TypeError, ValueError):
                wait_seconds = 60

            wait_seconds = max(wait_seconds, 1)

            logging.warning(
                "API вернул 429. Повтор запроса через %.1f сек.",
                wait_seconds,
            )

            response.close()
            time.sleep(wait_seconds)
            continue

        if response.status_code in (500, 502, 503, 504):
            server_error_retries += 1

            if server_error_retries > MAX_SERVER_ERROR_RETRIES:
                response.raise_for_status()

            wait_seconds = min(2 ** server_error_retries, 30)

            logging.warning(
                "Ошибка сервера HTTP %s. Повтор %s/%s через %s сек.",
                response.status_code,
                server_error_retries,
                MAX_SERVER_ERROR_RETRIES,
                wait_seconds,
            )

            response.close()
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        calls_left = response.headers.get("api_calls_left")
        rate_limit = response.headers.get("rate_limit_per_minute")

        if calls_left is not None:
            try:
                if int(calls_left) <= 2:
                    logging.info(
                        "Осталось запросов в лимите: %s; "
                        "лимит в минуту: %s",
                        calls_left,
                        rate_limit or "не указан",
                    )
            except ValueError:
                pass

        try:
            data = response.json()
        finally:
            response.close()

        time.sleep(REQUEST_DELAY)
        return data


# --------------------------------------------------
# Разбор ответов OmniDesk
# --------------------------------------------------

def extract_items(data, item_key):
    """
    Извлекает список объектов из ответа API.

    Поддерживает:
    - список объектов;
    - объект со списком по ключу item_key;
    - ответ с числовыми ключами "0", "1" и т. д.,
      где объекты могут быть вложены в case или message.
    """
    if isinstance(data, list):
        candidates = data

    elif isinstance(data, dict):
        if isinstance(data.get(item_key), list):
            candidates = data[item_key]
        else:
            candidates = [
                value
                for key, value in data.items()
                if key.isdigit() and isinstance(value, dict)
            ]

    else:
        return []

    result = []

    for item in candidates:
        if not isinstance(item, dict):
            continue

        nested = item.get(item_key)

        if isinstance(nested, dict):
            result.append(nested)
        elif item_key not in item:
            result.append(item)

    return result


def get_total_count(data):
    """Возвращает total_count, если API его передал."""
    if not isinstance(data, dict):
        return None

    value = data.get("total_count")

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------
# Выгрузка закрытых обращений
# --------------------------------------------------

def get_all_closed_cases():
    """Получает все закрытые обращения с пагинацией."""
    cases_by_id = {}
    page = 1
    total_count = None

    while True:
        data = get_json(
            "cases.json",
            params={
                "page": page,
                "limit": LIMIT,
                "status": "closed",
            },
        )

        items = extract_items(data, "case")

        if total_count is None:
            total_count = get_total_count(data)

        if not items:
            break

        added = 0

        for case in items:
            case_id = case.get("case_id")

            if case_id is None:
                continue

            status = str(case.get("status", "")).lower()

            if status != "closed":
                continue

            key = str(case_id)

            if key not in cases_by_id:
                cases_by_id[key] = case
                added += 1

        logging.info(
            "Страница обращений %s: получено %s, "
            "добавлено закрытых %s",
            page,
            len(items),
            added,
        )

        if len(items) < LIMIT:
            break

        if total_count is not None and page * LIMIT >= total_count:
            break

        if added == 0 and total_count is None:
            logging.warning(
                "На странице не появилось новых обращений; "
                "останавливаем пагинацию, чтобы избежать цикла."
            )
            break

        page += 1

    return list(cases_by_id.values())


# --------------------------------------------------
# Выгрузка переписки обращения
# --------------------------------------------------

def get_all_messages(case_id):
    """Получает все сообщения обращения в хронологическом порядке."""
    messages_by_id = {}
    messages_without_id = []
    page = 1
    total_count = None

    while True:
        data = get_json(
            f"cases/{case_id}/messages.json",
            params={
                "page": page,
                "limit": LIMIT,
                "order": "asc",
            },
        )

        items = extract_items(data, "message")

        if total_count is None:
            total_count = get_total_count(data)

        if not items:
            break

        for message in items:
            message_id = message.get("message_id")

            if message_id is None:
                messages_without_id.append(message)
            else:
                messages_by_id[str(message_id)] = message

        if len(items) < LIMIT:
            break

        if (
            total_count is not None
            and page * LIMIT >= total_count
        ):
            break

        page += 1

    messages = list(messages_by_id.values()) + messages_without_id

    # Сортируем по времени, сохраняя порядок API для сообщений
    # с одинаковым временем.
    messages.sort(
        key=lambda message: str(message.get("created_at") or "")
    )

    return messages


# --------------------------------------------------
# Нормализация сообщений
# --------------------------------------------------

def get_role(message):
    """Определяет тип автора сообщения."""
    note = message.get("note")

    if note is True or str(note).lower() == "true":
        return "internal_note"

    staff_id = message.get("staff_id")
    user_id = message.get("user_id")

    if staff_id not in (None, "", 0, "0"):
        return "support"

    if user_id not in (None, "", 0, "0"):
        return "customer"

    return "unknown"


def normalize_message(message):
    """Приводит сообщение к единому формату."""
    return {
        "message_id": message.get("message_id"),
        "role": get_role(message),
        "content": message.get("content") or "",
        "content_html": message.get("content_html") or "",
        "created_at": message.get("created_at"),
        "sent_at": message.get("sent_at"),
        "note": message.get("note", False),
        "staff_id": message.get("staff_id"),
        "user_id": message.get("user_id"),
        "attachments": message.get("attachments") or [],
    }


# --------------------------------------------------
# Сохранение ошибок
# --------------------------------------------------

def save_error(case_id, error):
    """Добавляет ошибку в JSONL, не стирая предыдущие записи."""
    record = {
        "case_id": case_id,
        "error": str(error),
    }

    with ERRORS_PATH.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# --------------------------------------------------
# Возобновление выгрузки
# --------------------------------------------------

def load_exported_ids():
    """Загружает ID обращений, которые уже записаны в датасет."""
    exported_ids = set()

    if not DATASET_PATH.exists():
        return exported_ids

    with DATASET_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(
                    f"Некорректный JSON в {DATASET_PATH}, "
                    f"строка {line_number}. "
                    "Сначала исправь файл, затем перезапусти скрипт."
                )

            case_id = record.get("case_id")

            if case_id is None:
                raise SystemExit(
                    f"В {DATASET_PATH}, строка {line_number}, "
                    "нет поля case_id."
                )

            exported_ids.add(str(case_id))

    return exported_ids


# --------------------------------------------------
# Основной процесс
# --------------------------------------------------

def main():
    exported_ids = load_exported_ids()

    logging.info(
        "Уже сохранено обращений: %s",
        len(exported_ids),
    )

    logging.info("Получаем список закрытых обращений...")
    cases = get_all_closed_cases()

    logging.info("Найдено закрытых обращений: %s", len(cases))

    success = 0
    failed = 0
    empty = 0
    total_messages = 0
    skipped = 0

    with DATASET_PATH.open("a", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            case_id = case.get("case_id")

            if case_id is None:
                logging.warning(
                    "Пропускаем обращение без case_id"
                )
                failed += 1
                continue

            case_key = str(case_id)

            if case_key in exported_ids:
                skipped += 1
                continue

            try:
                messages = get_all_messages(case_id)

                record = {
                    "case_id": case_id,
                    "subject": case.get("subject"),
                    "status": case.get("status"),
                    "created_at": case.get("created_at"),
                    "updated_at": case.get("updated_at"),
                    "category": case.get("category"),
                    "labels": case.get("labels", []),
                    "messages": [
                        normalize_message(message)
                        for message in messages
                    ],
                }

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                output.flush()

                # Добавляем ID только после успешной записи.
                exported_ids.add(case_key)

                success += 1
                total_messages += len(messages)

                if not messages:
                    empty += 1

                logging.info(
                    "[%s/%s] Обращение %s: сообщений %s",
                    index,
                    len(cases),
                    case_id,
                    len(messages),
                )

            except (
                requests.RequestException,
                ValueError,
                KeyError,
            ) as error:
                failed += 1
                save_error(case_id, error)

                logging.exception(
                    "Ошибка выгрузки обращения %s",
                    case_id,
                )

    report = {
        "closed_cases_found": len(cases),
        "cases_exported_this_run": success,
        "cases_already_exported": skipped,
        "cases_exported_total": len(exported_ids),
        "cases_failed_this_run": failed,
        "cases_without_messages_this_run": empty,
        "messages_exported_this_run": total_messages,
        "dataset_path": str(DATASET_PATH),
        "errors_path": str(ERRORS_PATH),
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info(
        "Текущий запуск завершён:\n%s",
        json.dumps(report, ensure_ascii=False, indent=2),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning(
            "Выгрузка прервана пользователем. "
            "Сохранённые обращения останутся в датасете. "
            "Запусти скрипт снова, чтобы продолжить."
        )
