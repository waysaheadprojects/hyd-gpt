import os
import asyncio
import time
import threading
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st
from io import BytesIO
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

import fitz  # PyMuPDF for PDF extraction
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === Setup ===
nest_asyncio.apply()
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

st.set_page_config(page_title="Retail Hyderabad Researcher", page_icon="🚀", layout="centered")

# ✅ White style
st.markdown("""
<style>
body {
  background-color: white;
  color: black;
}
.stApp {
  background-color: white;
}
input {
  background-color: #f5f5f5 !important;
  color: black !important;
  font-size: 1.2rem !important;
  border: 1px solid #ddd !important;
}
h1 {
  text-align: center;
  font-size: 4rem;
  color: #111;
  font-weight: 600;
}
h2, h3, h4, p, label {
  color: #111 !important;
}
.stAlert {
  background: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# === Fix GPTResearcher ===
original = agent_creator.extract_json_with_regex
agent_creator.extract_json_with_regex = lambda response: None if not response else original(response)

# === Core LLM ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()
tavily = TavilySearchResults(k=1)
INDEX_PATH = "./faiss_index"

# === Get Hyderabad fact ===
async def get_hyderabad_fact():
    """Fetch live Hyderabad retail fact"""
    q = "Give me one recent interesting fact about retail in Hyderabad or Inorbit Mall Hyderabad."
    try:
        res = tavily.invoke({"query": q})
        if res and "results" in res and res["results"]:
            return res["results"][0]["content"].strip()
    except Exception:
        pass
    return "No recent Hyderabad retail fact found."

# === Vector ===
if os.path.exists(INDEX_PATH):
    vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    st.title("🚀 Retail Hyderabad Researcher")
    uploaded_file = st.file_uploader("Upload your PDF to index", type=["pdf"])
    if uploaded_file:
        with st.spinner("Indexing..."):
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = [page.get_text().strip() for page in pdf_doc if page.get_text().strip()]
            docs = [Document(page_content=t) for t in pages]
            if docs:
                vs = FAISS.from_documents(docs, embeddings)
                vs.save_local(INDEX_PATH)
                st.success("✅ Vector store created! Please reload.")
            else:
                st.warning("No text found.")
    st.stop()

# === RAG Chain ===
def get_retriever_chain():
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_rag_chain(chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use ONLY this context. Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

async def vector_lookup(query: str) -> str:
    """Look up with context + fact."""
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector match."
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = await chain.ainvoke({"chat_history": [], "input": query, "context": context})
    fact = await get_hyderabad_fact()
    return f"{result['answer']}\n\n💡 **Hyderabad Retail Fact:** {fact}"

# === Deep Research ===
class CustomLogsHandler:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    async def send_json(self, data: Dict[str, Any]) -> None:
        self.logs.append(data)

async def run_gpt_researcher(query: str, logs_handler: CustomLogsHandler) -> str:
    """Deep Research with GPTResearcher."""
    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        report_source="hybrid",
        vector_store=vs,
        websocket=logs_handler
    )
    await researcher.conduct_research()
    report = await researcher.write_report()
    fact = await get_hyderabad_fact()
    return f"{report}\n\n💡 **Hyderabad Retail Fact:** {fact}"

def run_gpt_researcher_sync(query: str, logs_handler: CustomLogsHandler) -> str:
    return asyncio.get_event_loop().run_until_complete(run_gpt_researcher(query, logs_handler))

# === LangGraph ===
vector_tool = StructuredTool.from_function(vector_lookup)

class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state: State):
    res = await llm.ainvoke(f'Classify: "{state.query}". Return vector.')
    return {"route": "vector"}

async def vector_node(state: State):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.set_entry_point("router")
graph.add_edge("router", "vector")
graph.add_edge("vector", END)
agent = graph.compile()

# === UI ===
async def main():
    st.markdown(
        "<h1 style='text-align: center; font-size: 4rem;'>Retail Hyderabad Agent</h1>",
        unsafe_allow_html=True
    )

    # ✅ Show Hyderabad fact every time
    fact = await get_hyderabad_fact()
    st.info(f"💡 **Hyderabad Retail Fact:** {fact}")

    # === Compact input + buttons row ===
    col1, col2, col3 = st.columns([8, 1, 1])
    with col1:
        query = st.text_input(
            "",
            placeholder="Ask anything about Hyderabad retail...",
            label_visibility="collapsed",
            key="query_input"
        )
    with col2:
        send = st.button("➡️")
    with col3:
        run_deep = st.button("🚀")

    # === Logic: normal vector search ===
    if send and query:
        result = await agent.ainvoke(State(query=query))
        answer = result["answer"]
        if asyncio.iscoroutine(answer):
            answer = await answer
        st.write(answer)
        if answer.startswith("❌"):
            st.info("Try Deep Research instead!")

    # === Logic: Deep Research ===
    if run_deep:
        if not query.strip():
            st.warning("Please type a topic first.")
        else:
            def stream_research():
                logs_handler = CustomLogsHandler()
                result_holder = {"report": ""}

                def run_and_store():
                    result_holder["report"] = run_gpt_researcher_sync(query, logs_handler)

                t = threading.Thread(target=run_and_store)
                t.start()

                last_index = 0
                while t.is_alive():
                    time.sleep(1)
                    new_logs = logs_handler.logs[last_index:]
                    for log in new_logs:
                        yield f"🔍 **{log.get('content','')}**\n\n{log.get('output','')}\n\n---"
                    last_index += len(new_logs)

                final_report = result_holder["report"]
                yield f"\n\n## ✅ Final Report\n\n{final_report}"

                pdf_buffer = BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=LETTER)
                styles = getSampleStyleSheet()
                story = []
                for line in final_report.split("\n"):
                    if line.strip().startswith("# "):
                        story.append(Paragraph(f"<b>{line.strip('# ').strip()}</b>", styles["Heading1"]))
                    elif line.strip().startswith("## "):
                        story.append(Paragraph(f"<b>{line.strip('# ').strip()}</b>", styles["Heading2"]))
                    elif line.strip().startswith("### "):
                        story.append(Paragraph(f"<b>{line.strip('# ').strip()}</b>", styles["Heading3"]))
                    elif line.strip() == "---":
                        story.append(Spacer(1, 12))
                    else:
                        story.append(Paragraph(line, styles["BodyText"]))
                    story.append(Spacer(1, 6))

                doc.build(story)
                pdf_buffer.seek(0)
                st.download_button(
                    "📄 Download Report as PDF",
                    data=pdf_buffer,
                    file_name="deep_research.pdf",
                    mime="application/pdf"
                )

            st.write_stream(stream_research)


asyncio.run(main())
