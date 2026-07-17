from pathlib import Path
import os

from config.settings import setting
from tools.mcp_service.base_service import MCPServiceClient

class ArxivMCPClient(MCPServiceClient):
    def __init__(self, 
                command_or_url: str = "uv",
                args: list[str] | None = None,
                env: dict[str, str] | None = None,
                storage_path: str | Path = setting.pdf_save_dir,
                timeout: int = 100,
                sse_read_timeout: int = 300,
                auth=None,
                sampling_callback=None,
                headers=None,
                tool_call_logs_callback=None,
                http_client=None,
            ):
        if args is None:
            args = [
                "tool",
                "run",
                "arxiv-mcp-server",
                "--storage-path",
                str(storage_path),
            ]
        default_env = {
            **os.environ,
            "TESSDATA_PREFIX": "/usr/share/tesseract-ocr/5/tessdata",
        }
        if env is not None:
            default_env.update(env)

        super().__init__(
                        command_or_url=command_or_url,
                        args=args,
                        env=default_env,
                        timeout=timeout,
                        sse_read_timeout=sse_read_timeout,
                        auth=auth,
                        sampling_callback=sampling_callback,
                        headers=headers,
                        tool_call_logs_callback=tool_call_logs_callback,
                        http_client=http_client,
                    )

    async def search_papers(self,
                            query: str,
                            max_results: int | None = None,
                            **kwargs
        ):
        arguments = {
            "query": query,
            **kwargs,
        }
        if max_results is not None:
            arguments["max_results"] = max_results

        result = await self.call_tool("search_papers", arguments)
        return self.parse_result(result)

    async def download_paper(self, paper_id: str, **kwargs):
        result = await self.call_tool(
            "download_paper",
            {
                "paper_id": paper_id,
                **kwargs,
            },
        )
        return self.parse_result(result)
