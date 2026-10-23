"""Offline agent-loop tests using the real MCP SDK and an observed HTTP transport.

The action model is scripted and the desktop is a fixture. No LLM or sandbox
credentials are needed; only the network boundary of Search MCP is replaced.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import importlib
import json
import os
import threading
import shutil
import types
import unittest
from unittest.mock import patch
from uuid import UUID

import httpx

from os_computer_use.search import ParallelSearch, SEARCH_URL, USER_AGENT
from os_computer_use.llm_provider import OpenAIBaseProvider


# Avoid loading the configured GPU grounding service and credentialed LLM clients.
# Execute config.py itself so its default provider selection is still exercised.
providers = types.ModuleType("os_computer_use.providers")
providers.OSAtlasProvider = lambda: None
providers.OpenRouterProvider = lambda model: None
providers.GroqProvider = lambda model: None
with patch.dict(os.environ, {}, clear=True), patch.dict(
    "sys.modules", {"os_computer_use.providers": providers}
):
    config = importlib.import_module("os_computer_use.config")
    agent_module = importlib.import_module("os_computer_use.sandbox_agent")

AsyncClient = httpx.AsyncClient


class MCPServer:
    def __init__(self):
        self.requests = []
        self.calls = []
        self.error = False
        self.http_status = 200
        self.delay = False
        self.cancelled = threading.Event()

    async def handle(self, request):
        self.requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if self.http_status != 200:
            return httpx.Response(self.http_status, text="Explicit endpoint failure")
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            self.calls.append(message["params"])
            if self.delay:
                try:
                    await asyncio.sleep(60)
                finally:
                    self.cancelled.set()
            text = (
                "quota exceeded; retry later"
                if self.error
                else "E2B Desktop docs: https://e2b.dev/docs - desktop screenshot API"
            )
            result = {
                "content": [{"type": "text", "text": text}],
                "isError": self.error,
            }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": message["id"], "result": result}
        )

    def client(self, **kwargs):
        return AsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)


class Sandbox:
    def screenshot(self):
        with open("tests/test_screenshot.png", "rb") as image:
            return image.read()

    def set_timeout(self, timeout):
        pass


class ScriptedModel:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def call(self, messages, tools):
        self.seen.append((list(messages), tools))
        name, parameters = next(self.steps)
        return None, [{"name": name, "parameters": parameters}]


SEARCH = (
    "web_search",
    {"objective": "Find desktop docs", "query": "E2B desktop screenshot API"},
)
FETCH = ("web_fetch", {"url": "https://e2b.dev/docs"})
STOP = ("stop", {})


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.transport = patch(
            "os_computer_use.search.httpx.AsyncClient", self.server.client
        )
        self.transport.start()
        self.logger = patch.object(
            agent_module.logger, "log", side_effect=lambda text, *a, **kw: text
        )
        self.logger.start()
        self.vision = patch.object(
            agent_module,
            "vision_model",
            types.SimpleNamespace(call=lambda messages: "Research docs"),
        )
        self.vision.start()

    def tearDown(self):
        self.vision.stop()
        self.logger.stop()
        self.transport.stop()
        self.env.stop()

    def agent(self, **kwargs):
        agent = agent_module.SandboxAgent(Sandbox(), save_logs=False, **kwargs)
        self.addCleanup(shutil.rmtree, agent.tmp_dir)
        return agent

    def run_steps(self, agent, steps):
        model = ScriptedModel(steps)
        with patch.object(agent_module, "action_model", model):
            agent.run("Research desktop docs")
        return model

    def test_default_agent_roundtrip_result_consumption_and_followup(self):
        self.assertIsInstance(config.search_provider, ParallelSearch)
        agent = self.agent()
        model = self.run_steps(agent, [SEARCH, FETCH, STOP])
        self.assertIn("web_search", model.seen[0][1])
        self.assertIn("web_fetch", model.seen[0][1])
        self.assertIn("run_command", model.seen[0][1])
        self.assertIn("https://e2b.dev/docs", str(model.seen[1][0]))
        self.assertIn("desktop screenshot API", str(model.seen[2][0]))
        self.run_steps(agent, [SEARCH, STOP])
        calls = self.server.calls
        self.assertEqual(
            [c["name"] for c in calls], ["web_search", "web_fetch", "web_search"]
        )
        self.assertEqual(
            {c["arguments"]["session_id"] for c in calls}, {agent.search_session_id}
        )
        UUID(agent.search_session_id)
        self.assertEqual(calls[0]["arguments"]["search_queries"], [SEARCH[1]["query"]])
        self.assertEqual(calls[1]["arguments"]["urls"], ["https://e2b.dev/docs"])
        methods = set()
        for request in self.server.requests:
            self.assertEqual(str(request.url), SEARCH_URL)
            self.assertEqual(request.headers["User-Agent"], USER_AGENT)
            self.assertNotIn("authorization", request.headers)
            if request.method == "POST":
                methods.add(json.loads(request.content)["method"])
        self.assertTrue({"initialize", "tools/list", "tools/call"} <= methods)

    def test_shared_provider_interleaved_conversations_and_reset(self):
        provider = ParallelSearch.from_env()
        first, second = self.agent(search_provider=provider), self.agent(
            search_provider=provider
        )
        with ThreadPoolExecutor(2) as executor:
            results = list(
                executor.map(
                    lambda agent: agent.call_function(*SEARCH), (first, second)
                )
            )
        self.assertTrue(all("https://e2b.dev/docs" in r for r in results))
        first.call_function(*FETCH)
        second.call_function(*FETCH)
        ids = [c["arguments"]["session_id"] for c in self.server.calls]
        self.assertEqual(
            set(ids[:2]), {first.search_session_id, second.search_session_id}
        )
        self.assertNotEqual(first.search_session_id, second.search_session_id)
        self.assertEqual(ids[2:], [first.search_session_id, second.search_session_id])
        old = first.search_session_id
        first.messages.append({"role": "user", "content": "old memory"})
        first.reset_memory()
        self.assertEqual(first.messages, [])
        self.assertNotEqual(old, first.search_session_id)
        first.call_function(*SEARCH)
        self.assertEqual(
            self.server.calls[-1]["arguments"]["session_id"], first.search_session_id
        )

    def test_disabled_and_explicit_none_remove_tools_and_prevent_calls(self):
        for value in ("false", "0", "off", "no"):
            os.environ["WEB_SEARCH_ENABLED"] = value
            self.assertIsNone(ParallelSearch.from_env())
        agent = self.agent(search_provider=None)
        model = self.run_steps(agent, [SEARCH, STOP])
        self.assertNotIn("web_search", model.seen[0][1])
        self.assertNotIn("web_fetch", model.seen[0][1])
        self.assertEqual(agent.call_function(*FETCH), "Function not implemented.")
        self.assertIn("disabled", agent.web_search(**SEARCH[1]))
        self.assertEqual(self.server.requests, [])
        os.environ["WEB_SEARCH_ENABLED"] = "invalid"
        with self.assertRaises(ValueError):
            ParallelSearch.from_env()

    def test_explicit_credentials_and_endpoint_are_preserved(self):
        os.environ.update(
            PARALLEL_API_KEY="test-key",
            PARALLEL_SEARCH_URL="https://example.test/mcp?mode=fast",
        )
        agent = self.agent(search_provider=ParallelSearch.from_env())
        self.run_steps(agent, [SEARCH, STOP])
        self.assertTrue(self.server.requests)
        for request in self.server.requests:
            self.assertEqual(str(request.url), "https://example.test/mcp?mode=fast")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertEqual(request.headers["User-Agent"], USER_AGENT)

    def test_custom_provider_and_failures_do_not_fall_back(self):
        class Custom:
            def search(self, objective, query, session_id):
                raise PermissionError("custom search denied")

        custom = Custom()
        agent = self.agent(search_provider=custom)
        model = self.run_steps(agent, [SEARCH, STOP])
        self.assertIn("custom search denied", str(model.seen[-1][0]))
        self.assertIs(agent.search_provider, custom)
        self.assertEqual(self.server.requests, [])

    def test_mcp_errors_and_retries_keep_identity(self):
        agent = self.agent()
        self.server.error = True
        model = self.run_steps(agent, [SEARCH, STOP])
        self.assertIn("Search MCP error: quota exceeded", str(model.seen[-1][0]))
        self.server.error = False
        self.run_steps(agent, [SEARCH, STOP])
        self.assertEqual(
            {c["arguments"]["session_id"] for c in self.server.calls},
            {agent.search_session_id},
        )

    def test_http_failure_is_observed_without_anonymous_retry(self):
        self.server.http_status = 401
        agent = self.agent(search_provider=ParallelSearch(api_key="explicit-key"))
        model = self.run_steps(agent, [SEARCH, STOP])
        self.assertIn("Error executing function:", str(model.seen[-1][0]))
        posts = [r for r in self.server.requests if r.method == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].headers["authorization"], "Bearer explicit-key")

    def test_timeout_cancels_network_task_and_returns_observation(self):
        self.server.delay = True
        agent = self.agent(search_provider=ParallelSearch(timeout=0.15))
        model = self.run_steps(agent, [SEARCH, STOP])
        self.assertIn("Search MCP timed out after 0.15 seconds", str(model.seen[-1][0]))
        self.assertTrue(self.server.cancelled.wait(1))

    def test_keyboard_interrupt_cancels_portal_task(self):
        self.server.delay = True
        agent = self.agent()
        original = Future.result

        def interrupt(future, *args, **kwargs):
            # Interrupt the caller only once the actual MCP tool request is running.
            if threading.current_thread() is threading.main_thread():
                for _ in range(200):
                    if self.server.calls:
                        raise KeyboardInterrupt
                    threading.Event().wait(0.005)
            return original(future, *args, **kwargs)

        with patch.object(Future, "result", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                agent.call_function(*SEARCH)
        self.assertTrue(self.server.cancelled.wait(1))

    def test_invalid_fetch_urls_never_reach_transport(self):
        agent = self.agent()
        for url in (
            "file:///etc/passwd",
            "javascript:alert(1)",
            "https:///missing",
            "https://user:secret@example.com",
        ):
            result = agent.call_function("web_fetch", {"url": url})
            self.assertIn("Error executing function:", result)
        self.assertEqual(self.server.requests, [])

    def test_tool_permission_and_model_schema(self):
        agent = self.agent()
        provider = OpenAIBaseProvider.__new__(OpenAIBaseProvider)
        schemas = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in provider.create_function_schema(agent.tools)
        }
        self.assertEqual(schemas["web_search"]["required"], ["objective", "query"])
        self.assertEqual(schemas["web_fetch"]["properties"]["url"]["type"], "string")
        self.assertNotIn("session_id", schemas["web_search"]["properties"])
        del agent.tools["web_search"]
        self.assertEqual(agent.call_function(*SEARCH), "Function not implemented.")
        self.assertEqual(self.server.requests, [])


if __name__ == "__main__":
    unittest.main()
