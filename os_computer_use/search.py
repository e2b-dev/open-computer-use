"""Parallel Search MCP for the synchronous computer-use tool dispatcher."""

import os
from urllib.parse import urlsplit

import anyio
from anyio.from_thread import start_blocking_portal
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation


SEARCH_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "open-computer-use/0.1.0 (https://github.com/e2b-dev/open-computer-use)"


def validate_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Expected an HTTP or HTTPS URL without embedded credentials.")
    return url


class ParallelSearch:
    """A reusable provider; conversation IDs are supplied by the owning agent."""

    def __init__(self, url=SEARCH_URL, api_key=None, timeout=30):
        self.url = validate_url(url)
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        enabled = os.getenv("WEB_SEARCH_ENABLED", "true").strip().lower()
        if enabled in ("false", "0", "no", "off"):
            return None
        if enabled not in ("true", "1", "yes", "on"):
            raise ValueError("WEB_SEARCH_ENABLED must be true or false.")
        return cls(
            url=os.getenv("PARALLEL_SEARCH_URL", SEARCH_URL),
            api_key=os.getenv("PARALLEL_API_KEY") or None,
        )

    def search(self, objective, query, session_id):
        return self._call(
            "web_search",
            {
                "objective": objective,
                "search_queries": [query],
                "session_id": session_id,
            },
        )

    def fetch(self, url, session_id):
        return self._call(
            "web_fetch",
            {
                "urls": [validate_url(url)],
                "session_id": session_id,
            },
        )

    def _call(self, name, arguments):
        # main.start already runs an event loop. A short-lived AnyIO portal lets
        # this synchronous dispatcher use the SDK without nesting asyncio.run.
        # Cancel its task on Ctrl-C as well as on normal exit; no detached work.
        with start_blocking_portal() as portal:
            future = portal.start_task_soon(self._request, name, arguments)
            try:
                return future.result()
            except TimeoutError as error:
                raise TimeoutError(
                    f"Search MCP timed out after {self.timeout} seconds."
                ) from error
            finally:
                future.cancel()

    async def _request(self, name, arguments):
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with anyio.fail_after(self.timeout):
            async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as http:
                async with streamable_http_client(
                    self.url, http_client=http
                ) as streams:
                    async with ClientSession(
                        streams[0],
                        streams[1],
                        client_info=Implementation(
                            name="open-computer-use", version="0.1.0"
                        ),
                    ) as session:
                        await session.initialize()
                        available = await session.list_tools()
                        if name not in {tool.name for tool in available.tools}:
                            raise RuntimeError(f"Search MCP does not offer {name}.")
                        result = await session.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        if result.isError:
            return f"Search MCP error: {text or 'request failed'}"
        return text or "Search MCP returned no text content."
