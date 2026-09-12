import unittest


from scripts.training_env import parse_hf_identity, parse_ollama_models


class TestTrainingEnvironmentParsers(unittest.TestCase):
    def test_finds_exact_ollama_model_tag(self):
        body = '{"models":[{"name":"gemma3:4b"},{"name":"qwen2.5:14b"}]}'

        self.assertTrue(parse_ollama_models(body, "qwen2.5:14b"))
        self.assertFalse(parse_ollama_models(body, "qwen2.5:7b"))

    def test_accepts_expected_hugging_face_identity(self):
        self.assertTrue(parse_hf_identity("rajivmehtapy\n", "rajivmehtapy"))

    def test_accepts_hugging_face_cli_username_output(self):
        self.assertTrue(
            parse_hf_identity("username: rajivmehtapy\norgs:\n", "rajivmehtapy")
        )

    def test_accepts_current_hugging_face_user_equals_output(self):
        self.assertTrue(
            parse_hf_identity(
                "user=rajivmehtapy orgs=context-course,ml-intern-explorers\n",
                "rajivmehtapy",
            )
        )
        self.assertFalse(parse_hf_identity("someone-else\n", "rajivmehtapy"))

    def test_rejects_malformed_service_responses(self):
        self.assertFalse(parse_ollama_models("not-json", "qwen2.5:14b"))
        self.assertFalse(parse_hf_identity("", "rajivmehtapy"))


if __name__ == "__main__":
    unittest.main()
