from __future__ import annotations

from prompts.prompts import AnalyzerPrompt
from schema.analyzer_schema import QueryAnalysis
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate

from config.settings import setting


class QueryAnalyzer:
    def __init__(self, llm: OpenAILike = None):
        llm_model = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )

        self.llm = llm or llm_model

        self.prompt = PromptTemplate(AnalyzerPrompt)

    async def analyze(self, query: str) -> QueryAnalysis:
        result = await self.llm.astructured_predict(
            output_cls=QueryAnalysis,
            prompt=self.prompt,
            query=query,
        )
        
        return result
