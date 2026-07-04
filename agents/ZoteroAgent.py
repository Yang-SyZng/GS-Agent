from __future__ import annotations
import os

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
import logging

from prompts import ZoteroAgentSystemPrompt
from tools import ZoteroClientTools

logger = logging.getLogger(__name__)

def build_model() -> BaseChatModel:
    """Create the chat model"""
    from config import setting
    logger.info(f"正在构建LLM...")
    return ChatOpenAI(
        api_key=setting.LLM_API_KEY,
        base_url=setting.LLM_BASE_URL,
        model=setting.LLM_MODEL_ID,
    )

def ZoteroAgent(tools: list[any] = ZoteroClientTools):
    """Create agent"""
    model = build_model()
    logger.info(f"正在构建Zotero Agent...")
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=ZoteroAgentSystemPrompt,
    )