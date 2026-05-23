import asyncio
from typing import Any, AsyncGenerator

class MockStreamResult:
    def __init__(self, assistant_text: str = "", tool_calls: list = None):
        self.assistant_text = assistant_text
        self.tool_calls = tool_calls or []

class MockOpenAICompatClient:
    """
    Deterministically yields predefined chunks to simulate an LLM.
    Used for instant local testing of AHO and harness state mechanics.
    """
    def __init__(self, base_url: str = "", model: str = "", api_key: str = "", chat_endpoint: str = ""):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.chat_endpoint = chat_endpoint

    async def stream_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AsyncGenerator[dict[str, Any], None]:
        # Yield a fake thinking block and a fake tool call or text response
        yield {"choices": [{"delta": {"content": "<think>\nMock LLM deterministically selecting action\n</think>\n"}}]}
        await asyncio.sleep(0.01)

        # Decide whether to emit a tool based on input messages
        msg_str = str(messages)
        if "Find the current weather" in msg_str or tools:
            # Emit <tool_call>
            tool_xml = "<tool_call>\\n{\"name\": \"weather_lookup\", \"arguments\": {\"location\": \"Paris\"}}\\n</tool_call>"
            yield {"choices": [{"delta": {"content": tool_xml}}]}
        else:
            yield {"choices": [{"delta": {"content": "Task completed successfully inside mock."}}]}

    @classmethod
    def collect_stream(cls, chunks: list[dict]) -> MockStreamResult:
        text = ""
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text += delta.get("content", "")
        return MockStreamResult(assistant_text=text)
