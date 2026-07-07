from __future__ import annotations
import os

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import FunctionAgent

import logging

from prompts import ZoteroAgentSystemPrompt
from tools import ZoteroClientTools

logger = logging.getLogger(__name__)

def build_model() -> OpenAILike:
    """Create the chat model"""
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

def ZoteroAgent(tools: list[any] = ZoteroClientTools):
    """Create agent"""
    model = build_model()
    logger.info(f"正在构建Zotero Agent...")
    return FunctionAgent(
        name="ZoteroAgent",
        description="用于管理搜索处理Zotero的Agent",
        llm=model,
        tools=tools,
        system_prompt=ZoteroAgentSystemPrompt,
        verbose=True
    )