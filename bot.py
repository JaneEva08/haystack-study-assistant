"""Run: python bot.py. Private chats only; SQLite data stays outside Git."""
import logging
import os
import time
from pathlib import Path

from core import Store
from haystack_tutor import HaystackTutor
from pinecone_manager import PineconeManager
from transport import ChatClient, Telegram, load_env

ROOT = Path(__file__).resolve().parent
HELP = """Преподаватель алгебры · 7 класс. Сейчас изучаем линейные уравнения.
Haystack-агент: диалог и память, факты о кошках, анализ собак и погода.
Примеры: «Факт о кошках», «Покажи и опиши собаку», «Погода в Москве».

/lesson — начать объяснение (с очищенной историей)
/memory_on — включить память
/memory_off — отключить: прежняя история не используется, новая не сохраняется
/status — режим и число сохранённых сообщений
/reset — начать новый контекст
/memory_status — результат последней записи Pinecone
/help — команды

При включении памяти прежняя история возвращается. Для чистого сравнения
сначала используй /reset. Память отдельная для каждого Telegram-аккаунта.
"""
KEYBOARD = {"keyboard": [[{"text": "/lesson"}, {"text": "/status"}],
                         [{"text": "/memory_on"}, {"text": "/memory_off"}],
                         [{"text": "/reset"}, {"text": "/help"}]], "resize_keyboard": True}


def handle(message, tutor, telegram, allowed):
    if message.get("chat", {}).get("type") != "private" or message.get("from", {}).get("is_bot"):
        return
    user_id, chat_id = message["from"]["id"], message["chat"]["id"]
    if allowed and user_id not in allowed:
        telegram.send(chat_id, "Это семейный учебный бот. Доступ пока не настроен для этого аккаунта.")
        return
    text = message.get("text", "").strip()
    if not text:
        telegram.send(chat_id, "Пока работаю только с текстом. Документы добавим в следующем ДЗ.")
        return
    command = text.split()[0].split("@")[0].lower()
    if command in ("/start", "/help"):
        telegram.send(chat_id, HELP, KEYBOARD)
        return
    if command == "/memory_status":
        info = tutor.last_actions.get(user_id)
        telegram.send(chat_id, "Последняя операция памяти: " + (str(info) if info else "ещё нет записей в этом запуске"))
        return
    if command == "/status":
        enabled = tutor.store.memory_on(user_id)
        count = len(tutor.store.history(user_id)) * 2
        telegram.send(chat_id, f"Память {'включена' if enabled else 'выключена'}. "
                      f"В SQLite сохранено сообщений: {count}. "
                      "При выключенной памяти они не передаются модели.")
        return
    if command in ("/memory_on", "/memory_off"):
        enabled = command == "/memory_on"
        tutor.store.set_memory(user_id, enabled)
        telegram.send(chat_id, "Память включена. Сохранённая история снова используется." if enabled
                      else "Память выключена. История не используется и не пополняется. "
                           "Начать новый контекст — /reset.")
        return
    if command == "/reset":
        tutor.reset(user_id)
        telegram.send(chat_id, "Начат новый контекст. Прежние записи долгой памяти больше не используются.")
        return
    if command == "/lesson":
        tutor.reset(user_id)
        text = "Начни занятие по линейным уравнениям с небольшого объяснения и одного вопроса."
    elif text.startswith("/"):
        telegram.send(chat_id, "Неизвестная команда. Список: /help")
        return
    try:
        result = tutor.answer(user_id, text, original_text="/lesson" if command == "/lesson" else text)
    except (RuntimeError, ValueError) as exc:
        telegram.send(chat_id, str(exc))
        return
    if tutor.last_images:
        for photo in tutor.last_images:
            try:
                telegram.call("sendPhoto", chat_id=chat_id, photo=photo["url"], caption=photo["analysis"][:450])
            except RuntimeError:
                telegram.send(chat_id, "Telegram не смог получить картинку. Описание и ссылка будут в ответе ниже.")
    telegram.send(chat_id, result)


def main():
    load_env(ROOT / ".env")
    token, key = os.getenv("BOT_TOKEN", ""), os.getenv("OPENAI_API_KEY", "")
    if not token or not key or token.startswith("YOUR_") or key.startswith("YOUR_"):
        raise SystemExit("Заполни BOT_TOKEN и OPENAI_API_KEY в локальном .env по .env.example.")
    allowed = {int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
    store = Store(ROOT / "data" / "tutor.sqlite3")
    manager = PineconeManager(prefix="vpg06")
    tutor = HaystackTutor(store, (ROOT / "lesson.md").read_text(encoding="utf-8"), manager)
    telegram = Telegram(token)
    for attempt in range(4):
        try:
            me = telegram.call("getMe")
            webhook = telegram.call("getWebhookInfo")
            break
        except RuntimeError:
            if attempt == 3:
                raise
            logging.warning("Telegram temporarily unavailable during startup; retrying")
            time.sleep(2)
    if webhook.get("url"):
        raise SystemExit("У бота настроен webhook. Используй отдельного бота или сначала согласуй переключение.")
    store.bind_bot(me["id"])
    logging.info("Tutor polling started; stored context loads on each request")
    while True:
        try:
            updates = telegram.call("getUpdates", offset=store.offset(), timeout=25,
                                    allowed_updates=["message"])
            for update in updates:
                if "message" in update:
                    handle(update["message"], tutor, telegram, allowed)
                store.set_offset(update["update_id"] + 1)
        except RuntimeError as exc:
            logging.warning("%s", exc)
            time.sleep(3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        while True:
            try:
                main()
                break
            except RuntimeError as exc:
                message = str(exc)
                if any(code in message for code in ("HTTP 401", "HTTP 403", "HTTP 404")):
                    raise SystemExit("Проверь ключи и доступ сервисов в локальном .env.") from None
                logging.warning("Startup connection failed; retry in 20 seconds: %s", message)
                time.sleep(20)
    except KeyboardInterrupt:
        pass
