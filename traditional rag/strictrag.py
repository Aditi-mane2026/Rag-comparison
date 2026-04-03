from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_strict_rag(retriever, query: str):
    # Retrieve documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Strict prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized assistant. You MUST answer the user's query STRICTLY based on the provided context. "
                   "If the provided context does not contain the information to answer the query, "
                   "you must reply EXACTLY with: 'I cannot answer this based on the provided document.' "
                   "Do NOT use any external knowledge. Do NOT guess. Be extremely concise (1-2 sentences maximum)."),
        ("user", "Context:\n{context}\n\nQuery:\n{query}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "query": query})
    
    return {
        "answer": response.content,
        "docs": [doc.page_content for doc in docs]
    }
