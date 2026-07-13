from .state import AgentState
from agents.Analyzer import QueryAnalyzer
from agents.Retriever import PaperRetriever
from agents.Evaluator import RetrievalEvaluator
from agents.Researcher import ResearchSynthesizer

analyzer = QueryAnalyzer()
retriever = PaperRetriever()
evaluator = RetrievalEvaluator()
researcher = ResearchSynthesizer()

# analyzer node
async def analyzer_node(state: AgentState):
    analysis = await analyzer.analyze(state["query"])
    return {
        "analysis": analysis
    }

# retriever node
async def retriever_node(state: AgentState):
    nodes = await retriever.retrieve(analysis=state["analysis"])
    return {
        "retrieved_nodes": nodes
    }

# evaluator node
async def evaluator_node(state: AgentState):
    evaluation = await evaluator.evaluate(
                                    query=state["query"],
                                    analysis=state["analysis"],
                                    retrieved_nodes=state["retrieved_nodes"]
                                )
    return {
        "retrieval_evaluation": evaluation
    }

# synthesizer node
async def synthesizer_node(state: AgentState):
    research_result = await researcher.synthesis(
                                            query=state["query"],
                                            analysis=state["analysis"],
                                            retrieved_nodes=state["retrieved_nodes"],
                                            retrieval_evaluation=state["retrieval_evaluation"]
                                        )
    return{
        "research_result": research_result
    }