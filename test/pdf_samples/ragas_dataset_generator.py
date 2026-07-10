from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

from ragas.testset import TestsetGenerator

from config.settings import setting

llm = OpenAILike(
            model=setting.LLM_MODEL_ID,
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128000,
        )

embed = OpenAILikeEmbedding(
    model_name=setting.EMBEDDING_MODEL_ID,
    api_base=setting.BASE_URL,
    api_key=setting.API_KEY
)


documents = SimpleDirectoryReader(
    input_files=[
        "./database/parser/AbsGS/AbsGS.md"
    ]
).load_data()

generator = TestsetGenerator(llm=LlamaIndexLLMWrapper(llm), embedding_model=LlamaIndexEmbeddingsWrapper(embed))
dataset = generator.generate_with_llamaindex_docs(documents, testset_size=10)

print(dataset)