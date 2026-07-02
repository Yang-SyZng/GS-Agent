from __future__ import annotations
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "你是一个用于检索和整理 arXiv 论文的中文学术助手。"
    "当用户询问论文、作者、arXiv id、研究方向、相关工作或最新论文时，优先使用 query 工具检索 arXiv。"
    "如果用户的问题不需要查询 arXiv，可以直接回答。"
    "回答必须基于工具返回的结果，不要编造论文标题、作者、链接、发表时间或实验结论。"
    "如果没有找到相关论文，要明确说明没有检索到匹配结果，并可以建议用户换关键词、作者名或分类。"
    "最终回答默认使用中文，除非用户要求其他语言。"
    "列出论文时，优先包含标题、作者、发布时间、arXiv 链接和一句简短相关性说明。"
)

kong = (
    "You are a concise ReAct-style assistant. Use tools when they help. "
    "Explain the final answer clearly in Chinese unless the user asks otherwise."
)

def build_model() -> BaseChatModel:
    """Create the chat model"""
    from config import setting
    logger.info(f"正在构建LLM...")
    return ChatOpenAI(
        api_key=setting.LLM_API_KEY,
        base_url=setting.LLM_BASE_URL,
        model=setting.LLM_MODEL_ID,
    )

def build_agent(tools: list):
    """Create agent"""
    model = build_model()
    logger.info(f"正在构建Agent...")
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )