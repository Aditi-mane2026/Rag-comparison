from typing import List, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults

class GraphState(TypedDict):
    query: str
    documents: List[str]
    generation: str
    status: str

# Pydantic model for grading
class GradeDocuments(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def run_crag(retriever, query: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system_eval = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'. \n
    Otherwise, grade it as 'no'."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_eval),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    retrieval_grader = grade_prompt | structured_llm_grader

    # Define nodes
    def retrieve(state):
        question = state["query"]
        docs = retriever.invoke(question)
        return {"documents": [d.page_content for d in docs], "query": question}

    def evaluate_documents(state):
        question = state["query"]
        documents = state["documents"]
        
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke({"question": question, "document": d})
            if score.binary_score == "yes":
                filtered_docs.append(d)
                
        # CRAG Classification: correct, incorrect, ambiguous
        if len(filtered_docs) == len(documents) and len(documents) > 0:
            status = "correct"
            docs_to_keep = filtered_docs
        elif len(filtered_docs) == 0:
            status = "incorrect"
            docs_to_keep = [] # Discard all, rely on web search entirely
        else:
            status = "ambiguous"
            docs_to_keep = filtered_docs # Keep relevant ones, append web search next
            
        return {"documents": docs_to_keep, "query": question, "status": status}

    def web_search(state):
        question = state["query"]
        documents = state["documents"]
        
        web_tool = TavilySearchResults(k=2)
        try:
            docs = web_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            documents.append(f"Web Search Results:\n{web_results}")
        except Exception:
            documents.append("External search unavailable.")
            
        return {"documents": documents, "query": question}

    def generate(state):
        question = state["query"]
        documents = state["documents"]
        context = "\n\n".join(documents)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant. Answer the query carefully using the provided context. "
                       "If not available in context, state so. Keep your final answer extremely concise (1-2 sentences maximum)."),
            ("human", "Context:\n{context}\n\nQuery:\n{question}")
        ])
        chain = prompt | llm
        
        res = chain.invoke({"context": context, "question": question})
        return {"generation": res.content}

    def decide_to_generate(state):
        if state["status"] == "correct":
            return "generate"
        elif state["status"] == "incorrect":
            return "web_search"
        elif state["status"] == "ambiguous":
            return "web_search"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("evaluate", evaluate_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    result = app.invoke({"query": query})
    return {
        "answer": result["generation"],
        "docs": result["documents"]
    }
