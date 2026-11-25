import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import DeepSeekClient


class DeepSeekTests(unittest.TestCase):
    def test_total_budget_stops_next_call(self):
        client = DeepSeekClient(api_key="test-key", total_token_budget=10)
        client.usages.append({"total_tokens": 10})
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            client([{"role": "user", "content": "test"}])

    def test_request_uses_json_mode_and_model(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
        captured = []
        def fake_open(request, timeout):
            captured.append(json.loads(request.data))
            return Response(b'{"choices":[{"message":{"content":"{\\"action\\":\\"yield\\"}"}}]}')
        with patch("urllib.request.urlopen", fake_open):
            output = DeepSeekClient(api_key="test-key")([{"role": "user", "content": "test"}])
        self.assertIn('"yield"', output)
        self.assertEqual(captured[0]["model"], "deepseek-flash")
        self.assertEqual(captured[0]["response_format"]["type"], "json_object")


if __name__ == "__main__":
    unittest.main()
