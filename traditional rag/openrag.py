from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def run_open_rag(retriever, query: str):
    # Retrieve documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Open prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's query utilizing the provided context if it contains relevant information. "
                   "If the context is insufficient or out of subject, you are ALLOWED and ENCOURAGED to use your own broad knowledge base to answer the query. "
                   "Keep your final answer extremely concise (1-2 sentences maximum)."),
        ("user", "Context:\n{context}\n\nQuery:\n{query}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "query": query})
    
    return {
        "answer": response.content,
        "docs": [doc.page_content for doc in docs]
    }
