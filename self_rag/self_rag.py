from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class SelfRagState(TypedDict):
    query: str
    documents: List[str]
    generation: str
    needs_retrieval: bool
    is_relevant: bool
    is_supported: bool
    is_useful: bool
    loop_count: int
    revise_count: int

class GradeRetrieval(BaseModel):
    binary_score: str = Field(description="Query requires retrieval, 'yes' or 'no'")

class GradeRelevance(BaseModel):
    binary_score: str = Field(description="Documents are relevant, 'yes' or 'no'")

class GradeSupport(BaseModel):
    binary_score: str = Field(description="Answer is supported by documents, 'yes' or 'no'")

class GradeUtility(BaseModel):
    binary_score: str = Field(description="Answer is useful, 'yes' or 'no'")

def run_self_rag(retriever, query: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Graders
    decide_grader = llm.with_structured_output(GradeRetrieval)
    relevance_grader = llm.with_structured_output(GradeRelevance)
    support_grader = llm.with_structured_output(GradeSupport)
    utility_grader = llm.with_structured_output(GradeUtility)

    def decide_retrieval(state: SelfRagState):
        q = state["query"]
        decision = decide_grader.invoke([("system", "Does this query require looking up specific external documents to answer factually? Answer yes or no."), ("human", q)])
        return {"needs_retrieval": decision.binary_score == "yes"}

    def generate_direct(state: SelfRagState):
        q = state["query"]
        response = llm.invoke([("system", "Keep your final answer extremely concise (1-2 sentences maximum)."), ("human", q)])
        return {"generation": response.content}

    def retrieve(state: SelfRagState):
        q = state["query"]
        docs = retriever.invoke(q)
        return {"documents": [d.page_content for d in docs]}

    def check_relevance(state: SelfRagState):
        q = state["query"]
        docs = state["documents"]
        relevant = False
        for d in docs:
            score = relevance_grader.invoke([("system", "Are these documents relevant to the query? yes or no."), ("human", f"Query: {q}\nDoc: {d}")])
            if score.binary_score == "yes":
                relevant = True
                break
        return {"is_relevant": relevant}

    def generate_from_context(state: SelfRagState):
        q = state["query"]
        docs = state["documents"]
        context = "\n".join(docs)
        prompt = ChatPromptTemplate.from_messages([("system", "Answer based strictly on context. Keep your final answer extremely concise (1-2 sentences maximum)."), ("human", "Context:\n{context}\n\nQuery:\n{query}")])
        chain = prompt | llm
        res = chain.invoke({"context": context, "query": q})
        return {"generation": res.content, "revise_count": 0}

    def revise_answer(state: SelfRagState):
        q = state["query"]
        docs = state["documents"]
        prev_gen = state["generation"]
        context = "\n".join(docs)
        prompt = ChatPromptTemplate.from_messages([("system", "Revise the answer to be strictly supported by the context. Keep your final answer extremely concise (1-2 sentences maximum)."), ("human", f"Context:\n{context}\n\nQuery:\n{q}\n\nPrevious Answer:\n{prev_gen}\n\nPlease revise:")])
        chain = prompt | llm
        res = chain.invoke({})
        return {"generation": res.content, "revise_count": state.get("revise_count", 0) + 1}

    def check_support(state: SelfRagState):
        q = state["query"]
        docs = state["documents"]
        gen = state["generation"]
        context = "\n".join(docs)
        score = support_grader.invoke([("system", "Is the answer fully supported by context? yes or no."), ("human", f"Context: {context}\nAnswer: {gen}")])
        return {"is_supported": score.binary_score == "yes"}

    def check_utility(state: SelfRagState):
        q = state["query"]
        gen = state["generation"]
        score = utility_grader.invoke([("system", "Does the answer usefully resolve the query? yes or no."), ("human", f"Query: {q}\nAnswer: {gen}")])
        return {"is_useful": score.binary_score == "yes"}

    def rewrite_query(state: SelfRagState):
        q = state["query"]
        prompt = ChatPromptTemplate.from_messages([("system", "Rewrite this query to be better for semantic search."), ("human", q)])
        res = (prompt | llm).invoke({"query": q})
        return {"query": res.content, "loop_count": state.get("loop_count", 0) + 1}

    def no_answer_found(state: SelfRagState):
        return {"generation": "Insufficient information available to answer the query reliably."}

    # Graph construction
    workflow = StateGraph(SelfRagState)
    workflow.add_node("decide_retrieval", decide_retrieval)
    workflow.add_node("generate_direct", generate_direct)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("generate_from_context", generate_from_context)
    workflow.add_node("revise_answer", revise_answer)
    workflow.add_node("check_support", check_support)
    workflow.add_node("check_utility", check_utility)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("no_answer_found", no_answer_found)

    workflow.add_edge(START, "decide_retrieval")

    def route_decision(state: SelfRagState):
        return "retrieve" if state["needs_retrieval"] else "generate_direct"
    
    workflow.add_conditional_edges("decide_retrieval", route_decision, {"retrieve": "retrieve", "generate_direct": "generate_direct"})
    workflow.add_edge("generate_direct", END)
    
    workflow.add_edge("retrieve", "check_relevance")

    def route_relevance(state: SelfRagState):
        if state["is_relevant"]:
            return "generate_from_context"
        else:
            if state.get("loop_count", 0) >= 2: # Keep the loop limit tight
                return "no_answer_found"
            return "rewrite_query"
    
    workflow.add_conditional_edges("check_relevance", route_relevance, {"generate_from_context": "generate_from_context", "rewrite_query": "rewrite_query", "no_answer_found": "no_answer_found"})
    
    workflow.add_edge("generate_from_context", "check_support")
    workflow.add_edge("revise_answer", "check_support")

    def route_support(state: SelfRagState):
        if state["is_supported"]:
            return "check_utility"
        elif state.get("revise_count", 0) < 2:
            return "revise_answer"
        else:
            return "check_utility" # After max revisions, just check utility anyway
    
    workflow.add_conditional_edges("check_support", route_support, {"check_utility": "check_utility", "revise_answer": "revise_answer"})

    def route_utility(state: SelfRagState):
        if state["is_useful"]:
            return END
        else:
            if state.get("loop_count", 0) >= 2:
                return "no_answer_found"
            return "rewrite_query"
            
    workflow.add_conditional_edges("check_utility", route_utility, {END: END, "rewrite_query": "rewrite_query", "no_answer_found": "no_answer_found"})
    
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("no_answer_found", END)

    app = workflow.compile()
    result = app.invoke({"query": query, "loop_count": 0, "revise_count": 0, "documents": [], "generation": ""})
    return {
        "answer": result["generation"],
        "docs": result.get("documents", [])
    }
