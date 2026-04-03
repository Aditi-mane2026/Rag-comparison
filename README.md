# Advanced RAG Intelligence Dashboard

A comprehensive platform designed to compare and evaluate four distinct Retrieval-Augmented Generation (RAG) architectures. This project demonstrates the differences in grounding, reasoning, and self-correction across various RAG implementations.

## 🚀 RAG Architectures

### 1. Traditional (Strict) RAG
- **Behavior:** Strictly grounded in the provided document.
- **Goal:** To prevent hallucinations by only answering if the answer is explicitly found in the retrieved context.
- **Outcome:** High precision, but may fail to answer valid queries if the exact wording isn't found.

### 2. Open RAG
- **Behavior:** More conversational and flexible with the retrieved information.
- **Goal:** Provides a helpful summary even if the context is broad.
- **Outcome:** Better user experience for general queries, but higher risk of slight hallucinations.

### 3. Corrective RAG (CRAG)
- **Behavior:** Evaluates the quality of retrieved documents.
- **Goal:** If retrieval is insufficient or irrelevant, it triggers an external web search (via Tavily) to augment the context.
- **Outcome:** Robust against "knowledge gaps" in the local vector store.

### 4. Self-Reflective RAG (Self-RAG)
- **Behavior:** An iterative state-machine flow with self-critique.
- **Workflow:** 
    - **Decide Retrieval:** Determines if retrieval is even needed.
    - **Relevance Check:** Grades retrieved docs for actual usefulness.
    - **Self-Support Check:** Verifies the generated answer against the context.
    - **Utility Check:** Final check for answer completeness.
    - **Revision Loop:** Rewrites queries or revises answers if thresholds aren't met (with bounded iteration limits).
- **Outcome:** The most reliable and detailed output due to continuous self-correction.

## 🛠️ Technology Stack
- **Backend:** FastAPI, LangGraph (for complex RAG flows)
- **AI Framework:** LangChain, OpenAI (GPT-4o-mini)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend:** Vanilla HTML5, CSS3 (Modern, premium aesthetic)
- **Tools:** Tavily (for CRAG web search)

## 📦 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install fastapi uvicorn langchain langchain-openai faiss-cpu pydantic python-dotenv PyPDF2 langgraph tavily-python
    ```

3.  **Configure Environment:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key_here
    TAVILY_API_KEY=your_tavily_key_here
    ```

4.  **Run the Application:**
    ```bash
    python app.py
    ```

5.  **Access the Dashboard:**
    Open `http://127.0.0.1:8000` in your web browser.

## 📂 Project Structure
- `app.py`: Main FastAPI server and UI bridge.
- `index.html`: Interactive comparison dashboard.
- `shared/`: Common utilities (Vector store builder).
- `traditional rag/`: Strict and Open RAG implementations.
- `C-rag/`: Corrective RAG logic.
- `self_rag/`: Self-Reflective RAG graph logic.
- `apple_fruit_confusion_doc_v2.pdf`: Default knowledge base.

---
*Created with focus on RAG transparency and performance benchmarking.*
