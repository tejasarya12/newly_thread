"""
Graph Builder (Legacy/Batch Mode)
"""
from langgraph.graph import StateGraph, START, END
from nodes import SupportState, input_router

def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("input_router", input_router)
    graph.add_edge(START, "input_router")
    graph.add_edge("input_router", END)
    return graph.compile()















"""
from langgraph.graph import StateGraph, START, END
from .nodes import (
    SupportState,
    input_router,
    intent_analyzer,
    retriever_node,
    response_generator,
    accuracy_evaluator,
    output_router,
    check_accuracy_route
)

def build_graph():
  
    # Initialize state graph
    graph = StateGraph(SupportState)
    
    # Add all nodes
    graph.add_node("input_router", input_router)
    graph.add_node("intent_analyzer", intent_analyzer)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("response_generator", response_generator)
    graph.add_node("accuracy_evaluator", accuracy_evaluator)
    graph.add_node("output_router", output_router)
    
    # Add edges (linear flow with one conditional)
    graph.add_edge(START, "input_router")
    graph.add_edge("input_router", "intent_analyzer")
    graph.add_edge("intent_analyzer", "retriever_node")
    graph.add_edge("retriever_node", "response_generator")
    graph.add_edge("response_generator", "accuracy_evaluator")
    
    # Conditional edge based on accuracy
    graph.add_conditional_edges(
        "accuracy_evaluator",
        check_accuracy_route,
        {
            "output_router": "output_router",
            "retriever_node": "retriever_node"  # Loop back for refinement
        }
    )
    
    graph.add_edge("output_router", END)
    
    # Compile and return
    compiled_graph = graph.compile()
    
    print("✅ LangGraph workflow compiled successfully")
    return compiled_graph

def visualize_graph():
   
    try:
        graph = build_graph()
        # graph.get_graph().draw_png("workflow_diagram.png")
        print("Graph visualization saved as workflow_diagram.png")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    # Test the graph build
    graph = build_graph()
    print("\nTesting graph with sample state...")
    
    test_state = {
        "user_input": "How do I reset my device?",
        "mode_input": "text",
        "mode_output": "text",
        "intent": "",
        "retrieved_docs": [],
        "answer": "",
        "accuracy": False,
        "email": "",
        "audio_file": None,
        "confidence_score": 0.0
    }
    
    result = graph.invoke(test_state)
    print("\n📊 Final State:")
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Confidence: {result['confidence_score']}")