"""
تست‌های ModelGate — سیاست serial/concurrent بدون نیاز به mlx/torch واقعی
اجرا:  python tests/test_model_gate.py
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model_gate import ModelGate


class FakeModel:
    """مدل ساختگی برای شمارش لود/آنلود"""

    def __init__(self, name: str):
        self.name = name


class TestModelGateSerial(unittest.TestCase):
    def setUp(self):
        self.gate = ModelGate()
        # resident=[] → سخت‌گیرانه‌ترین حالت serial (هیچ مدلی مستثنا نیست)
        self.gate.configure({"mode": "serial", "log_memory": False, "resident": []})
        self.loads = {"a": 0, "b": 0}

    def _loader(self, name):
        def _fn():
            self.loads[name] += 1
            return FakeModel(name)
        return _fn

    @staticmethod
    def _resident_keys(gate):
        return [e["key"] for e in gate.stats()["loaded"]]

    def test_cache_hit_no_reload(self):
        m1 = self.gate.acquire("a", "llm", self._loader("a"))
        m2 = self.gate.acquire("a", "llm", self._loader("a"))
        self.assertIs(m1, m2)
        self.assertEqual(self.loads["a"], 1)

    def test_serial_evicts_other_models(self):
        self.gate.acquire("a", "llm", self._loader("a"))
        self.gate.acquire("b", "llm", self._loader("b"))
        # در حالت serial فقط آخرین مدل باید رزیدنت باشد
        self.assertEqual(self._resident_keys(self.gate), ["b"])

        # برگشت به a → لود مجدد
        self.gate.acquire("a", "llm", self._loader("a"))
        self.assertEqual(self.loads["a"], 2)
        self.assertEqual(self._resident_keys(self.gate), ["a"])

    def test_resident_kind_survives_serial_eviction(self):
        """kind های resident (مثل embedding) در حالت serial هم آزاد نمی‌شوند"""
        self.gate.configure({"mode": "serial", "resident": ["embedding"]})
        self.gate.acquire("emb", "embedding", lambda: FakeModel("emb"))
        self.gate.acquire("llm1", "llm", lambda: FakeModel("llm1"))
        self.assertEqual(sorted(self._resident_keys(self.gate)), ["emb", "llm1"])

        # لود LLM دوم: llm1 آزاد می‌شود ولی embedding می‌ماند
        self.gate.acquire("llm2", "llm", lambda: FakeModel("llm2"))
        self.assertEqual(sorted(self._resident_keys(self.gate)), ["emb", "llm2"])

    def test_unloader_called_on_evict(self):
        unloaded = []
        self.gate.acquire("a", "llm", self._loader("a"), unloader=lambda: unloaded.append("a"))
        self.gate.acquire("b", "llm", self._loader("b"))
        self.assertEqual(unloaded, ["a"])

    def test_serial_slot_is_exclusive(self):
        self.gate.acquire("a", "llm", self._loader("a"))
        active = []
        max_active = []

        def worker():
            with self.gate.execution_slot("a"):
                active.append(1)
                max_active.append(len(active))
                time.sleep(0.02)
                active.pop()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(max(max_active), 1)

    def test_acquire_waits_for_inflight_inference(self):
        """مدل جدید نباید وسط inference مدل قبلی لود شود (ریشه‌ی METAL OOM)"""
        self.gate.acquire("a", "llm", self._loader("a"))
        order = []

        def infer():
            with self.gate.execution_slot("a"):
                order.append("infer_start")
                time.sleep(0.05)
                order.append("infer_end")

        def load_other():
            time.sleep(0.01)  # مطمئن شویم inference شروع شده
            self.gate.acquire("b", "llm", self._loader("b"))
            order.append("b_loaded")

        t1 = threading.Thread(target=infer)
        t2 = threading.Thread(target=load_other)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(order, ["infer_start", "infer_end", "b_loaded"])

    def test_evict(self):
        self.gate.acquire("a", "llm", self._loader("a"))
        self.assertTrue(self.gate.evict("a"))
        self.assertFalse(self.gate.evict("a"))
        self.assertEqual(self.gate.stats()["loaded"], [])

    def test_evict_all_respects_resident(self):
        self.gate.configure({"resident": ["embedding"]})
        self.gate.acquire("emb", "embedding", lambda: FakeModel("emb"))
        self.gate.acquire("llm1", "llm", lambda: FakeModel("llm1"))

        self.gate.evict_all()
        self.assertEqual(self._resident_keys(self.gate), ["emb"])

        self.gate.evict_all(include_resident=True)
        self.assertEqual(self.gate.stats()["loaded"], [])

    def test_acquire_inside_slot_no_deadlock(self):
        """الگوی واقعی کلاینت‌ها: acquire داخل execution_slot از چند thread
        با کلیدهای متفاوت — نباید deadlock شود (ترتیب قفل slot → lock)"""
        done = []

        def worker(key):
            for _ in range(10):
                with self.gate.execution_slot(key):
                    self.gate.acquire(key, "llm", lambda: FakeModel(key))
            done.append(key)

        threads = [
            threading.Thread(target=worker, args=("a",)),
            threading.Thread(target=worker, args=("b",)),
            threading.Thread(target=worker, args=("c",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(sorted(done), ["a", "b", "c"], "deadlock یا timeout رخ داد")


class TestModelGateConcurrent(unittest.TestCase):
    def setUp(self):
        self.gate = ModelGate()
        self.gate.configure({
            "mode": "concurrent",
            "max_concurrent_inferences": 2,
            "log_memory": False,
        })

    def test_models_stay_resident(self):
        self.gate.acquire("a", "llm", lambda: FakeModel("a"))
        self.gate.acquire("b", "embedding", lambda: FakeModel("b"))
        keys = sorted(e["key"] for e in self.gate.stats()["loaded"])
        self.assertEqual(keys, ["a", "b"])

    def test_concurrency_bounded_by_semaphore(self):
        self.gate.acquire("a", "llm", lambda: FakeModel("a"))
        active = []
        max_active = []

        def worker():
            with self.gate.execution_slot("a"):
                active.append(1)
                max_active.append(len(active))
                time.sleep(0.02)
                active.pop()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(max(max_active), 2)
        self.assertGreaterEqual(max(max_active), 2)  # واقعاً موازی شد


class TestModeSwitch(unittest.TestCase):
    def test_switch_to_serial_keeps_most_recent(self):
        gate = ModelGate()
        gate.configure({"mode": "concurrent", "log_memory": False, "resident": []})
        gate.acquire("a", "llm", lambda: FakeModel("a"))
        time.sleep(0.01)
        gate.acquire("b", "llm", lambda: FakeModel("b"))
        gate.configure({"mode": "serial"})
        keys = [e["key"] for e in gate.stats()["loaded"]]
        self.assertEqual(keys, ["b"])

    def test_env_var_overrides_config(self):
        os.environ["MODEL_EXECUTION_MODE"] = "concurrent"
        try:
            gate = ModelGate()
            gate.configure({"mode": "serial"})  # باید توسط env بی‌اثر شود
            self.assertEqual(gate.mode, "concurrent")
        finally:
            del os.environ["MODEL_EXECUTION_MODE"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
