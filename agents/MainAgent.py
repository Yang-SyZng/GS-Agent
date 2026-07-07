from __future__ import annotations
import os
from typing import Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow import AgentStream
import logging

logger = logging.getLogger(__name__)

class BaseFunctionAgent(FunctionAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        if "llm" not in kwargs and len(args) <= 5:
            kwargs["llm"] = self.BuildModel()

        super().__init__(*args, **kwargs)
        logger.info(f"正在构建 {self.name} Agent...")
        
    def BuildModel(self,
    ) -> OpenAILike:
        """Create the CHAT&Function Calling model"""
        from config import setting
        logger.info(f"正在构建LLM...")
        return OpenAILike(
            model=setting.LLM_MODEL_ID,
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128000,
        )

    async def stream_run(self, msg):
        response = self.run(user_msg="11234 * 4567")
        async for event in response.stream_events():
            if isinstance(event, ToolCall):
                logger.info(f"Tool Call: {event.tool_name}, {event.tool_kwargs}")
            elif isinstance(event, ToolCallResult):
                logger.info(f"Tool Result:\n{event.tool_output}")
            started = False
            if isinstance(event, AgentStream):
                delta = event.delta
                if not started:
                    delta = delta.lstrip()
                    started = True
                print(
                    delta,
                    end="",
                    flush=True
                )