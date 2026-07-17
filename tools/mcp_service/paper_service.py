import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.settings import setting
from tools.mcp_service.base_service import MCPServiceClient


class PaperMCPClient(MCPServiceClient):
    def __init__(
        self,
        command_or_url: str = "uv",
        args: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        download_dir: str | Path = setting.pdf_save_dir,
        **kwargs: Any,
    ) -> None:
        self.download_dir = Path(download_dir).expanduser().resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if args is None:
            args = ["tool", "run", "paper-search-mcp"]

        process_env = os.environ.copy()
        if env is not None:
            process_env.update({key: str(value) for key, value in env.items()})

        super().__init__(
            command_or_url=command_or_url,
            args=args,
            env=process_env,
            **kwargs,
        )

    async def search(
        self,
        query: str,
        max_results_per_source: int = 5,
        sources: str | Sequence[str] = "all",
        year: str | None = None,
    ) -> Any:
        if not query.strip():
            raise ValueError("query 不能为空")
        if max_results_per_source < 1:
            raise ValueError("max_results_per_source 必须大于 0")

        source_value = (
            sources
            if isinstance(sources, str)
            else ",".join(source.strip() for source in sources if source.strip())
        )
        arguments: dict[str, Any] = {
            "query": query,
            "max_results_per_source": max_results_per_source,
            "sources": source_value or "all",
        }
        if year is not None:
            arguments["year"] = year

        result = await self.call_tool("search_papers", arguments)
        return self.parse_result(result)

    async def search_source(
        self,
        source: str,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> Any:
        normalized_source = source.strip().lower()
        if not normalized_source or not query.strip():
            raise ValueError("source 和 query 不能为空")

        result = await self.call_tool(
            f"search_{normalized_source}",
            {"query": query, "max_results": max_results, **kwargs},
        )
        return self.parse_result(result)

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> Any:
        """Search papers from arXiv only."""
        if not query.strip():
            raise ValueError("query 不能为空")
        if max_results < 1:
            raise ValueError("max_results 必须大于 0")

        result = await self.call_tool(
            "search_arxiv",
            {"query": query, "max_results": max_results, **kwargs},
        )
        return self.parse_result(result)

    async def download_arxiv(
        self,
        paper_id: str,
        save_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Download an arXiv paper to the configured download directory."""
        if not paper_id.strip():
            raise ValueError("paper_id 不能为空")

        target_dir = Path(save_path or self.download_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        result = await self.call_tool(
            "download_arxiv",
            {
                "paper_id": paper_id,
                "save_path": str(target_dir),
                **kwargs,
            },
        )
        return self.parse_result(result)

    async def read_arxiv(
        self,
        paper_id: str,
        save_path: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """Read an arXiv paper, downloading it first when required by the server."""
        if not paper_id.strip():
            raise ValueError("paper_id 不能为空")

        target_dir = Path(save_path or self.download_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        result = await self.call_tool(
            "read_arxiv_paper",
            {
                "paper_id": paper_id,
                "save_path": str(target_dir),
                **kwargs,
            },
        )
        return self.parse_result(result)

    async def download(
        self,
        source: str,
        paper_id: str,
        *,
        doi: str = "",
        title: str = "",
        save_path: str | Path | None = None,
        use_scihub: bool = False,
        scihub_base_url: str = "https://sci-hub.se",
    ) -> Any:
        if not source.strip() or not paper_id.strip():
            raise ValueError("source 和 paper_id 不能为空")

        target_dir = Path(save_path or self.download_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        result = await self.call_tool(
            "download_with_fallback",
            {
                "source": source,
                "paper_id": paper_id,
                "doi": doi,
                "title": title,
                "save_path": str(target_dir),
                "use_scihub": use_scihub,
                "scihub_base_url": scihub_base_url,
            },
        )
        return self.parse_result(result)

    async def read(
        self,
        source: str,
        paper_id: str,
        save_path: str | Path | None = None,
    ) -> Any:
        normalized_source = source.strip().lower()
        if not normalized_source or not paper_id.strip():
            raise ValueError("source 和 paper_id 不能为空")

        target_dir = Path(save_path or self.download_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        result = await self.call_tool(
            f"read_{normalized_source}_paper",
            {"paper_id": paper_id, "save_path": str(target_dir)},
        )
        return self.parse_result(result)
