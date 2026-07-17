from urllib.parse import urlencode

from config.settings import setting
from tools.mcp_service.base_service import MCPServiceClient


class TavilyMCPClient(MCPServiceClient):
    def __init__(self,
                command_or_url: str | None = None,
                api_key: str | None = setting.TAVILY_API_KEY,
                **kwargs
        ):
        if command_or_url is None:
            if not api_key:
                raise ValueError("TAVILY_API_KEY 未配置")

            query = urlencode({"tavilyApiKey": api_key})
            command_or_url = f"https://mcp.tavily.com/mcp/?{query}"

        super().__init__(
                        command_or_url=command_or_url,
                        **kwargs,
                    )

    async def search(self,
                     query: str,
                     **kwargs
        ):
        result = await self.call_tool(
            "tavily_search",
            {
                "query": query,
                **kwargs,
            },
        )
        return self.parse_result(result)

    async def extract(self, urls: str | list[str], **kwargs):
        result = await self.call_tool(
            "tavily_extract",
            {
                "urls": urls,
                **kwargs,
            },
        )
        return self.parse_result(result)
