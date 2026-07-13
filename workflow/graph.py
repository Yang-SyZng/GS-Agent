from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import analyzer_node, retriever_node, evaluator_node, synthesizer_node

builder = StateGraph(AgentState)

builder.add_node("analyzer", analyzer_node)
builder.add_node("retriever", retriever_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "analyzer")
builder.add_edge("analyzer", "retriever")
builder.add_edge("retriever", "evaluator")
builder.add_edge("evaluator", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()
