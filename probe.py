import json
import os
import tempfile
import uuid
from pathlib import Path
from transport import load_env
from core import Store
from pinecone_manager import PineconeManager
from haystack_tutor import HaystackTutor

ROOT = Path(__file__).resolve().parent


def main():
    load_env(ROOT / ".env")
    manager = PineconeManager(prefix="vpg06-check-" + uuid.uuid4().hex[:8])
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder) / "probe.sqlite3")
        tutor = HaystackTutor(store, (ROOT / "lesson.md").read_text(encoding="utf-8"), manager)
        results = []
        for prompt, tool in [("Получи один факт о кошках через API.", "get_cat_fact"),
                             ("Покажи случайную собаку и опиши, что видно на изображении.", "analyze_dog_image"),
                             ("Какая сейчас погода в Москве?", "get_weather")]:
            answer = tutor.answer(-601, prompt)
            trace = tutor.last_trace
            if tool not in [c["name"] for c in trace["tool_calls"]]:
                raise RuntimeError("Модель не вызвала нужный инструмент.")
            for r in trace["tool_results"]:
                value = json.loads(r["result"]) if isinstance(r["result"], str) else r["result"]
                if r["error"] or isinstance(value, dict) and "error" in value:
                    raise RuntimeError("Инструмент вернул ошибку: " + str(value))
            results.append({"prompt": prompt, **trace, "memory_action": tutor.last_actions[-601]})
            print(json.dumps(results[-1], ensure_ascii=False, indent=2), flush=True)
        tutor.answer(-602, "Запомни мою цель: на этой неделе научиться решать линейные уравнения.")
        store.reset(-602)
        answer = tutor.answer(-602, "Какую учебную цель я называла на эту неделю?")
        memory = {"after_short_history_clear": answer, "trace": tutor.last_trace}
        if "уравнен" not in answer.lower():
            raise RuntimeError("Восстановление цели требует проверки.")
        print(json.dumps(memory, ensure_ascii=False, indent=2), flush=True)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results/service-checks.json").write_text(json.dumps({"checks": results, "memory": memory},
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
