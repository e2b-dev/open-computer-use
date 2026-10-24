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
    models = (
        (None, "MiniMax-M3.1-Flash"),
        ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
        ("MiniMax-M3.1-Flash", "MiniMax-M3.1-Flash"),
        ("minimax-m3", "MiniMax-M3"),
        ("MiniMax-M3", "MiniMax-M3"),
        ("minimax-m2.7", "MiniMax-M2.7"),
        ("MiniMax-M2.7", "MiniMax-M2.7"),
        ("custom-model-id", "custom-model-id"),
    )
    tools = {
        "click_item": {
            "description": "Click on an item on the screen",
            "params": {"description": "Item to click"},
        }
    }

    def test_aliases_advertise_flash_first_and_m3_only(self):
        expected = [
            ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
            ("minimax-m3", "MiniMax-M3"),
        ]
        for provider in (MiniMaxProvider, MiniMaxAnthropicProvider):
            self.assertEqual(list(provider.aliases.items()), expected)

    def exercise_request(self, provider_class, base_url, model, functions=None,
                         completion_kwargs=None):
        seen = []
        image_data = Path(__file__).with_name("test_screenshot.png").read_bytes()
        messages = [
            Message("Keep the conversation context.", role="system"),
            Message("Previous reply"),
            Message(["Click the item", image_data], role="user"),
        ]
        anthropic = provider_class is MiniMaxAnthropicProvider

        def handler(request):
            body = json.loads(request.content)
            seen.append((str(request.url), body))
            if anthropic:
                content = [{"type": "text", "text": "ok"}]
                if functions:
                    content.append({
                        "type": "tool_use", "id": "tool_test", "name": "click_item",
                        "input": {"description": "item"},
                    })
                response = {
                    "id": "msg_test", "type": "message", "role": "assistant",
                    "model": body["model"], "content": content,
                    "stop_reason": "tool_use" if functions else "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            else:
                message = {"role": "assistant", "content": "ok"}
                if functions:
                    message["tool_calls"] = [{
                        "id": "tool_test", "type": "function",
                        "function": {
                            "name": "click_item",
                            "arguments": '{"description": "item"}',
                        },
                    }]
                response = {
                    "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if functions else "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2},
                }
            return Response(200, json=response)

        sdk_class = Anthropic if anthropic else OpenAI
        sdk_name = "Anthropic" if anthropic else "OpenAI"
        with Client(transport=MockTransport(handler)) as http_client:
            # Keep the real provider constructor and SDK serializer in the path.
            def create_sdk(**kwargs):
                return sdk_class(**kwargs, http_client=http_client, max_retries=0)

            with (
                patch.object(provider_class, "base_url", base_url),
                patch.object(provider_class, "api_key", "test"),
                patch("os_computer_use.llm_provider." + sdk_name,
                      side_effect=create_sdk),
            ):
                provider = provider_class() if model is None else provider_class(model)
                if completion_kwargs is not None:
                    provider.completion([Message("hello", role="user")],
                                        **completion_kwargs)
                else:
                    result = provider.call(messages, functions)
                    if functions:
                        self.assertEqual(result, ("ok", [{
                            "type": "function", "name": "click_item",
                            "parameters": {"description": "item"},
                        }]))
                    else:
                        self.assertEqual(result, "ok")

        self.assertEqual(len(seen), 1)
        url, body = seen[0]
        suffix = "/v1/messages" if anthropic else "/chat/completions"
        self.assertEqual(url, base_url + suffix)
        if completion_kwargs is None:
            self.assertEqual(body["messages"][-2], messages[1])
            content = body["messages"][-1]["content"]
            self.assertEqual(content[0], {"type": "text", "text": "Click the item"})
            if anthropic:
                self.assertEqual(body["system"], messages[0]["content"])
                self.assertEqual(body["max_tokens"], 4096)
                self.assertEqual(content[1]["source"]["media_type"], "image/png")
                encoded = content[1]["source"]["data"]
            else:
                self.assertEqual(body["messages"][0], messages[0])
                url = content[1]["image_url"]["url"]
                self.assertTrue(url.startswith("data:image/png;base64,"))
                encoded = url.split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), image_data)
            if functions:
                tool = body["tools"][0]
                if not anthropic:
                    tool = tool["function"]
                self.assertEqual(tool["name"], "click_item")
                schema = tool["input_schema" if anthropic else "parameters"]
                self.assertEqual(schema["required"], ["description"])
            else:
                self.assertNotIn("tools", body)
        return body

    def test_default_and_explicit_models_at_sdk_http_boundary(self):
        for provider_class, prefix in (
            (MiniMaxProvider, "/v1"),
            (MiniMaxAnthropicProvider, "/anthropic"),
        ):
            for host in ("api.minimax.io", "api.minimaxi.com"):
                for model, expected_model in self.models:
                    for functions in (None, self.tools):
                        with self.subTest(provider=provider_class.__name__, host=host,
                                          model=model, tools=bool(functions)):
                            body = self.exercise_request(
                                provider_class, "https://" + host + prefix,
                                model, functions,
                            )
                            self.assertEqual(body["model"], expected_model)
                            # Preserve server reasoning defaults and omit unsupported flags.
                            for field in ("thinking", "reasoning_effort", "output_config",
                                          "service_tier", "reasoning_split"):
                                self.assertNotIn(field, body)

    def test_custom_endpoint_and_model_are_preserved(self):
        for provider_class, base_url in (
            (MiniMaxProvider, "https://proxy.example/custom/v1"),
            (MiniMaxAnthropicProvider, "https://proxy.example/custom/anthropic"),
        ):
            with self.subTest(provider=provider_class.__name__):
                body = self.exercise_request(provider_class, base_url, "saved-custom-model")
                self.assertEqual(body["model"], "saved-custom-model")

    def test_explicit_flash_reasoning_options_reach_the_sdk_transport(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                body = self.exercise_request(
                    MiniMaxProvider, "https://api.minimax.io/v1", None,
                    completion_kwargs={"reasoning_effort": effort},
                )
                self.assertEqual(body["reasoning_effort"], effort)
                body = self.exercise_request(
                    MiniMaxAnthropicProvider, "https://api.minimax.io/anthropic", None,
                    completion_kwargs={
                        "max_tokens": 4096,
                        "extra_body": {"thinking": {"type": "adaptive"},
                                       "output_config": {"effort": effort}},
                    },
                )
                self.assertEqual(body["thinking"], {"type": "adaptive"})
                self.assertEqual(body["output_config"], {"effort": effort})


if __name__ == "__main__":
    unittest.main()
