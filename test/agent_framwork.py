from llama_index.llms.openai_like import OpenAILike
from llama_index.core.base.llms.types import ChatMessage
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentStream
from config import setting
from typing import Any
import logging
from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
)

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


llm = OpenAILike(
    model=setting.LLM_MODEL_ID,
    api_base=setting.BASE_URL,
    is_chat_model=True,
    is_function_calling_model=True,
    api_key=setting.API_KEY,
    context_window=128000,
)



# for response in llm.stream_chat(messages):
#     print(response.delta, end="", flush=True)

# print()


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


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

async def main():
    # Run the agent
    agent = BaseFunctionAgent(
        name="乘法",
        tools=[multiply],
        system_prompt="你是一个专门计算乘法的助手"
    )
    await agent.stream_run("12412 * 231451等于多少？")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())