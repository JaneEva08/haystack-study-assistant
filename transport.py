"""Small JSON HTTP adapters. Error messages never contain tokens or request bodies."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def post_json(url, body, headers=None, timeout=90):
    request = Request(url, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Сервис вернул HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Сервис недоступен или истекло время ожидания.") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("Сервис вернул некорректный JSON.") from None


class ChatClient:
    def __init__(self, api_key, base_url, model):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        if not self.base_url.startswith("https://"):
            raise ValueError("Для API необходим HTTPS URL.")

    def __call__(self, messages):
        result = post_json(self.base_url + "/chat/completions",
                           {"model": self.model, "messages": messages},
                           {"Authorization": "Bearer " + self.api_key})
        try:
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError()
            return answer
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("API не вернул текст ответа.") from None


class Telegram:
    def __init__(self, token):
        self.token = token

    def call(self, method, **payload):
        result = post_json(f"https://api.telegram.org/bot{self.token}/{method}", payload, timeout=40)
        if not result.get("ok"):
            raise RuntimeError("Telegram отклонил запрос.")
        return result["result"]

    def send(self, chat_id, text, keyboard=None):
        # UTF-16 length can exceed Python's len() for emojis; 1800 is conservative.
        for start in range(0, len(text), 1800):
            payload = {"chat_id": chat_id, "text": text[start:start + 1800]}
            if start == 0 and keyboard:
                payload["reply_markup"] = keyboard
            self.call("sendMessage", **payload)
