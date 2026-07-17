import json
from typing import Any

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


class MCPServiceClient(BasicMCPClient):

    @property
    def tool_spec(self) -> McpToolSpec:
        return McpToolSpec(client=self)

    async def get_tool_names(self) -> list[str]:
        result = await self.list_tools()
        return [tool.name for tool in result.tools]

    @staticmethod
    def parse_result(result) -> Any:
        if getattr(result, "isError", False):
            messages = [
                item.text
                for item in result.content
                if hasattr(item, "text")
            ]
            raise RuntimeError("\n".join(messages) or "MCP 工具调用失败")

        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            return structured_content

        text_contents = [
            item.text
            for item in result.content
            if hasattr(item, "text")
        ]

        if not text_contents:
            return result.content

        if len(text_contents) > 1:
            return [MCPServiceClient._parse_text(text) for text in text_contents]

        return MCPServiceClient._parse_text(text_contents[0])

    @staticmethod
    def _parse_text(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
