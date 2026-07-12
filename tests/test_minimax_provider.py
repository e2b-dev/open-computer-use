import base64
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from anthropic import Anthropic
from httpx import Client, MockTransport, Response
from openai import OpenAI

from os_computer_use.llm_provider import Message
from os_computer_use.providers import MiniMaxAnthropicProvider, MiniMaxProvider


class MiniMaxProviderTests(unittest.TestCase):
    def test_aliases_cover_supported_models(self):
        expected = {
            "minimax-m3": "MiniMax-M3",
            "minimax-m2.7": "MiniMax-M2.7",
        }
        self.assertEqual(MiniMaxProvider.aliases, expected)
        self.assertEqual(MiniMaxAnthropicProvider.aliases, expected)

    def test_openai_endpoints_append_chat_completions_once(self):
        endpoints = (
            ("https://api.minimax.io/v1", "/v1/chat/completions"),
            ("https://api.minimaxi.com/v1", "/v1/chat/completions"),
        )

        for base_url, expected_path in endpoints:
            with self.subTest(base_url=base_url):
                seen_paths = []

                def handler(request):
                    seen_paths.append(request.url.path)
                    return Response(
                        200,
                        json={
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "MiniMax-M3",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "ok",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )

                sdk_client = OpenAI(
                    api_key="test",
                    base_url=base_url,
                    http_client=Client(transport=MockTransport(handler)),
                )
                try:
                    with (
                        patch.object(MiniMaxProvider, "base_url", base_url),
                        patch(
                            "os_computer_use.llm_provider.OpenAI",
                            return_value=sdk_client,
                        ),
                    ):
                        provider = MiniMaxProvider("minimax-m3")
                        self.assertEqual(
                            provider.call([Message("hello", role="user")]), "ok"
                        )
                finally:
                    sdk_client.close()

                self.assertEqual(seen_paths, [expected_path])

    def test_anthropic_endpoints_append_v1_messages_once(self):
        endpoints = (
            ("https://api.minimax.io/anthropic", "/anthropic/v1/messages"),
            ("https://api.minimaxi.com/anthropic", "/anthropic/v1/messages"),
        )
        image_data = Path(__file__).with_name("test_screenshot.png").read_bytes()

        for base_url, expected_path in endpoints:
            with self.subTest(base_url=base_url):
                seen_requests = []

                def handler(request):
                    seen_requests.append(
                        (request.url.path, json.loads(request.content))
                    )
                    return Response(
                        200,
                        json={
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "model": "MiniMax-M3",
                            "content": [{"type": "text", "text": "ok"}],
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    )

                sdk_client = Anthropic(
                    api_key="test",
                    base_url=base_url,
                    http_client=Client(transport=MockTransport(handler)),
                )
                try:
                    with (
                        patch.object(MiniMaxAnthropicProvider, "base_url", base_url),
                        patch(
                            "os_computer_use.llm_provider.Anthropic",
                            return_value=sdk_client,
                        ),
                    ):
                        provider = MiniMaxAnthropicProvider("minimax-m3")
                        self.assertEqual(
                            provider.call([Message([image_data], role="user")]), "ok"
                        )
                finally:
                    sdk_client.close()

                self.assertEqual(len(seen_requests), 1)
                path, body = seen_requests[0]
                self.assertEqual(path, expected_path)
                self.assertEqual(path.count("/v1/messages"), 1)
                encoded_image = body["messages"][0]["content"][0]["source"]["data"]
                self.assertIsInstance(encoded_image, str)
                self.assertEqual(base64.b64decode(encoded_image), image_data)


if __name__ == "__main__":
    unittest.main()
