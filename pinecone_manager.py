"""Pinecone user-message memory; no assistant answers are stored as facts."""
import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from pinecone_client import Pinecone, DIMENSION, MODEL
from transport import load_env, ChatClient

ROOT = Path(__file__).resolve().parent


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def protected_signature(text):
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & {"не", "нет", "никогда", "без"}), tuple(re.findall(r"-?\d+(?:[.,]\d+)?", text))


def can_skip(new, old, score, threshold):
    return score >= threshold and protected_signature(new) == protected_signature(old)


class PineconeManager:
    def __init__(self, prefix="vpg05"):
        load_env(ROOT / ".env")
        self.pc = Pinecone(os.environ["PINECONE_API_KEY"])
        self.openai_base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.chat = ChatClient(os.getenv("OPENAI_API_KEY", ""), self.openai_base_url,
                               os.getenv("CHAT_MODEL", "gpt-6-luna"))
        self.threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))
        if not 0 < self.threshold <= 1:
            raise ValueError("Порог должен быть больше 0 и не больше 1.")
        index = self.pc.request("https://api.pinecone.io", "/indexes/" + os.getenv("PINECONE_INDEX", "algebra-tutor-search"))
        if index["dimension"] != DIMENSION or index["metric"] != "cosine":
            raise ValueError("Нужен индекс 1024/cosine.")
        self.host, self.prefix = "https://" + index["host"].removeprefix("https://"), prefix

    def complete(self, messages):
        return self.chat(messages)

    def namespace(self, user_id, epoch="initial"):
        return f"{self.prefix}-{user_id}-{epoch}"

    def query(self, vector, user_id, epoch="initial", extra_filter=None):
        filters = {"user_id": {"$eq": str(user_id)}, "type": {"$eq": "user_message"}}
        if extra_filter:
            filters.update(extra_filter)
        return self.pc.request(self.host, "/query", {"namespace": self.namespace(user_id, epoch),
            "vector": vector, "topK": 5, "includeMetadata": True, "includeValues": False,
            "filter": filters}).get("matches", [])

    def search(self, user_id, text, epoch="initial"):
        vector = self.pc.embed([text], "query")[0][0]
        return [{"id": x["id"], "score": x["score"], "text": x["metadata"]["text"]}
                for x in self.query(vector, user_id, epoch)]

    def remember(self, user_id, text, profile=None, epoch="initial"):
        if not text.strip() or len(text) > 3000:
            raise ValueError("Непустой исходный текст, не более 3000 символов.")
        if text.lstrip().startswith("/"):
            return {"action": "skipped", "reason": "command"}
        vector = self.pc.embed([text], "passage")[0][0]
        candidates = self.query(vector, user_id, epoch)
        for hit in candidates:
            old = hit["metadata"]["text"]
            if can_skip(text, old, hit["score"], self.threshold):
                result = {"action": "skipped", "id": hit["id"], "score": hit["score"], "threshold": self.threshold}
                logging.info("memory %s", json.dumps(result))
                return result
        id_ = "message-" + hashlib.sha256(text.encode()).hexdigest()[:24]
        metadata = {"id": id_, "user_id": str(user_id), "type": "user_message", "text": text,
                    "embedding_model": MODEL}
        # Only optional non-null names are accepted. No bot_response or generated summary.
        for key in ("first_name", "last_name", "username"):
            value = (profile or {}).get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value
        result = self.pc.request(self.host, "/vectors/upsert", {"namespace": self.namespace(user_id, epoch),
                  "vectors": [{"id": id_, "values": vector, "metadata": metadata}]})
        if result.get("upsertedCount") != 1:
            raise RuntimeError("Pinecone не подтвердил запись.")
        # Wait for search visibility so immediately following paraphrases can be compared.
        visible = False
        for _ in range(15):
            if self.query(vector, user_id, epoch, {"id": {"$eq": id_}}):
                visible = True
                break
            time.sleep(1)
        result = {"action": "inserted", "id": id_, "search_visible": visible,
                  "closest_score": candidates[0]["score"] if candidates else None, "threshold": self.threshold}
        logging.info("memory %s", json.dumps(result))
        return result


def probe():
    manager = PineconeManager(prefix="vpg05-check-" + uuid.uuid4().hex[:8])
    phrases = ["Мне трудно раскрывать скобки.", "Я испытываю трудности при раскрытии скобок.",
               "Я не испытываю трудности при раскрытии скобок.", "Мне нравится строить графики функций."]
    results = []
    for text in phrases:
        result = manager.remember("test-student", text, {"first_name": None, "last_name": None})
        results.append({"text": text, **result})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if results[0]["action"] != "inserted" or results[1]["action"] != "skipped":
        raise RuntimeError("Проверка смыслового повтора не прошла; порог требует анализа.")
    if results[2]["action"] != "inserted":
        raise RuntimeError("Отрицание ошибочно признано повтором.")
    other = manager.search("other-student", "Мне трудно раскрывать скобки.")
    report = {"namespace": manager.namespace("test-student"), "threshold": manager.threshold,
              "results": results, "other_student_matches": other}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/manager-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    probe()
