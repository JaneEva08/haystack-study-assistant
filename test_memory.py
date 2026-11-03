import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from core import Store
from pinecone_manager import PineconeManager, can_skip
from memory_tutor import MemoryTutor


class MemoryTests(unittest.TestCase):
    def manager(self, candidates):
        m = PineconeManager.__new__(PineconeManager)
        m.threshold, m.prefix, m.host = .95, "test", "https://test.pinecone.io"
        m.pc = Mock()
        m.pc.embed.return_value = ([[0.1] * 1024], {})
        m.pc.request.return_value = {"upsertedCount": 1}
        m.query = Mock(side_effect=[candidates, [{"id": "visible"}]])
        return m

    def test_similarity_threshold_and_guards(self):
        self.assertTrue(can_skip("Мне трудно раскрывать скобки", "Я испытываю трудности со скобками", .96, .95))
        self.assertFalse(can_skip("Я не понял скобки", "Я понял скобки", .999, .95))
        self.assertFalse(can_skip("x = 3", "x = 4", .999, .95))
        self.assertFalse(can_skip("Пример", "Пример", .94, .95))

    def test_duplicate_not_written(self):
        m = self.manager([{"id": "existing", "score": .97, "metadata": {"text": "Трудно раскрывать скобки"}}])
        self.assertEqual(m.remember(1, "Трудно раскрывать скобки")["action"], "skipped")
        m.pc.request.assert_not_called()

    def test_raw_text_and_missing_names(self):
        m = self.manager([])
        raw = "Мне трудно раскрывать скобки."
        result = m.remember(1, raw, {"first_name": None, "last_name": None, "bot_response": "generated"})
        self.assertEqual(result["action"], "inserted")
        metadata = m.pc.request.call_args.args[2]["vectors"][0]["metadata"]
        self.assertEqual(metadata["text"], raw)
        self.assertEqual(metadata["type"], "user_message")
        self.assertNotIn("first_name", metadata)
        self.assertNotIn("last_name", metadata)
        self.assertNotIn("bot_response", metadata)

    def test_commands_are_not_embedded(self):
        m = self.manager([])
        m.remember(1, "/start")
        m.pc.embed.assert_not_called()

    def test_user_and_epoch_namespaces(self):
        m = self.manager([])
        self.assertNotEqual(m.namespace(1), m.namespace(2))
        self.assertNotEqual(m.namespace(1), m.namespace(1, "new"))

    def test_modes_reset_and_original_command(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.db")
            manager = Mock()
            manager.search.return_value = []
            manager.remember.return_value = {"action": "inserted"}
            tutor = MemoryTutor(store, "Lesson", lambda _: "Ответ", manager)
            store.set_memory(1, False)
            tutor.answer(1, "Вопрос")
            manager.search.assert_not_called()
            manager.remember.assert_not_called()
            store.set_memory(1, True)
            tutor.answer(1, "Начни занятие", original_text="/lesson")
            self.assertEqual(manager.remember.call_args.args[1], "/lesson")
            old_epoch = tutor.epoch(1)
            tutor.reset(1)
            self.assertNotEqual(old_epoch, tutor.epoch(1))
            self.assertEqual(store.history(1), [])

    def test_failed_write_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.db")
            manager = Mock()
            manager.search.return_value = []
            manager.remember.side_effect = RuntimeError("Network")
            tutor = MemoryTutor(store, "Lesson", lambda _: "Ответ", manager)
            self.assertIn("не сохранила", tutor.answer(1, "Вопрос"))
            self.assertEqual(len(store.history(1)), 1)

    def test_new_bot_resets_polling_offset_only(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.db")
            store.append(1, "Вопрос", "Ответ")
            store.bind_bot(100)
            store.set_offset(900)
            store.bind_bot(100)
            self.assertEqual(store.offset(), 900)
            store.bind_bot(200)
            self.assertEqual(store.offset(), 0)
            self.assertEqual(len(store.history(1)), 1)


if __name__ == "__main__":
    unittest.main()
