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
from reportlab.platypus import Table, TableStyle

from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === Setup ===
nest_asyncio.apply()
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

st.set_page_config(page_title="Retail Agent", page_icon="🧩", layout="centered")

# ✅ White style + chat look
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
  font-weight: 700;
}
h2, h3, h4, p, label {
  color: #111 !important;
}
.stAlert {
  background: #f0f0f0;
}
.user-msg {
  background: #e0f7fa;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
}
.bot-msg {
  background: #f1f8e9;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# === Fix GPTResearcher ===
original = agent_creator.extract_json_with_regex
agent_creator.extract_json_with_regex = lambda response: None if not response else original(response)

# === Core LLM ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()
INDEX_PATH = "./faiss_index"

# === Get Hyderabad fact ===
tavily = TavilySearchResults(k=5)

async def get_latest_retail_news() -> str:
    """Fetch live news about Hyderabad retail, Inorbit Mall."""
    query = (
        "latest Hyderabad retail news OR Inorbit Mall site:business-standard.com "
        "OR site:economictimes.indiatimes.com OR site:newsmeter.in"
    )
    results = await tavily.ainvoke({"query": query})
    print(results)  # Debug

    if results and "results" in results and results["results"]:
        items = []
        for r in results["results"]:
            title = r.get("title", "No title")
            url = r.get("url", "#")
            content = r.get("content", "")
            items.append(f"🔗 **{title}**\n{content}\n👉 [Read more]({url})\n")
        return "\n\n".join(items)

    # fallback
    prompt = "Give me 2 interesting facts about Hyderabad retail with numbers and give just the intuitive fact."
    fallback = await llm.ainvoke(prompt)
    return fallback.content.strip()

# === Vector ===
if os.path.exists(INDEX_PATH):
    vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    st.title("🚀 Retail Hyderabad Agent")
    uploaded_file = st.file_uploader("Upload your PDF to index", type=["pdf"])
    if uploaded_file:
      with st.spinner("Indexing..."):
          # === Save uploaded file to my-docs ===
          os.makedirs("my-docs", exist_ok=True)
          save_path = os.path.join("my-docs", uploaded_file.name)
          with open(save_path, "wb") as f:
              f.write(uploaded_file.getbuffer())
  
          # === Read text for vector ===
          pdf_bytes = uploaded_file.read()
          pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
          pages = [page.get_text().strip() for page in pdf_doc if page.get_text().strip()]
          docs = [Document(page_content=t) for t in pages]
          if docs:
              vs = FAISS.from_documents(docs, embeddings)
              vs.save_local(INDEX_PATH)
              st.success(f"✅ Vector store created & file saved to my-docs! Reload to use it.")
          else:
              st.warning("No text found.")


# === RAG Chain with fallback ===
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
    """Try vector store, fallback to LLM if no match."""
    docs = vs.similarity_search(query, k=5)
    if not docs:
        fallback = await llm.ainvoke(f"User said: \"{query}\". Reply from general knowledge politely.")
        fact = await get_latest_retail_news()
        return f"{fallback.content.strip()}\n\n💡 **Hyderabad Retail Fact:** {fact}"
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = await chain.ainvoke({"chat_history": [], "input": query, "context": context})
    fact = await get_latest_retail_news()
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
    fact = await get_latest_retail_news()
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

import asyncio
import time
import threading
from io import BytesIO

import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet


async def main():
    # === Title ===
    st.markdown("<h1>Retail Agent</h1>", unsafe_allow_html=True)

    # === Fresh fact on each load ===
    fact = await get_latest_retail_news()
    st.info(f"💡 **Retail Fact:** {fact}")

    # === Init session state ===
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "deep_running" not in st.session_state:
        st.session_state.deep_running = False

    # === Render chat history ===
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # === Input & buttons ===
    col1, col2, col3 = st.columns([8, 1, 1])
    with col1:
        query = st.text_input(
            "",
            placeholder="Ask anything about retail...",
            label_visibility="collapsed",
            key="query_input"
        )
    with col2:
        send = st.button("➡️")
    with col3:
        run_deep = st.button("🚀")

    # === Handle normal ➡️ query ===
    if send and query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            result = await agent.ainvoke(State(query=query))
            answer = result["answer"]
            if asyncio.iscoroutine(answer):
                answer = await answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # === Handle 🚀 Deep Research trigger ===
    if run_deep:
        if not query.strip():
            st.warning("Enter a topic first.")
        else:
            st.session_state.deep_running = True

    # === Run Deep Research with live logs ===
    if st.session_state.deep_running:
        def stream_research():
            logs_handler = CustomLogsHandler()
            result_holder = {"report": ""}

            def run_and_store():
                result_holder["report"] = run_gpt_researcher_sync(query, logs_handler)

            t = threading.Thread(target=run_and_store)
            t.start()

            last_index = 0
            logs_placeholder = st.empty()

            while t.is_alive():
                time.sleep(1)
                new_logs = logs_handler.logs[last_index:]
                for log in new_logs:
                    logs_placeholder.markdown(
                        f"🔍 **{log.get('content','')}**\n\n```\n{log.get('output','')}\n```",
                        unsafe_allow_html=True
                    )
                last_index += len(new_logs)

            final_report = result_holder["report"]
            yield f"\n\n## ✅ Final Report\n\n{final_report}"

            # === Build stylish PDF ===
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=LETTER)
            styles = getSampleStyleSheet()
            story = []

            for line in final_report.split("\n"):
                line = line.strip()
                if not line:
                    continue
                elif line.startswith("# "):
                    story.append(Paragraph(f"<b>{line.strip('# ').strip()}</b>", styles["Heading1"]))
                elif line.startswith("## "):
                    story.append(Paragraph(f"<b>{line.strip('# ').strip()}</b>", styles["Heading2"]))
                elif "|" in line:
                    cols = [c.strip() for c in line.split("|") if c.strip()]
                    if not hasattr(stream_research, "_table_buffer"):
                        stream_research._table_buffer = []
                    stream_research._table_buffer.append(cols)
                else:
                    if hasattr(stream_research, "_table_buffer") and stream_research._table_buffer:
                        table = Table(stream_research._table_buffer)
                        table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), "#d0d0d0"),
                            ("GRID", (0, 0), (-1, -1), 1, "black"),
                            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ]))
                        story.append(table)
                        story.append(Spacer(1, 12))
                        stream_research._table_buffer = []
                    story.append(Paragraph(f"<b>{line}</b>", styles["BodyText"]))
                    story.append(Spacer(1, 6))

            if hasattr(stream_research, "_table_buffer") and stream_research._table_buffer:
                table = Table(stream_research._table_buffer)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), "#d0d0d0"),
                    ("GRID", (0, 0), (-1, -1), 1, "black"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ]))
                story.append(table)
                story.append(Spacer(1, 12))
                stream_research._table_buffer = []

            doc.build(story)
            pdf_buffer.seek(0)

            st.download_button(
                "📄 Download Stylish Report as PDF",
                data=pdf_buffer,
                file_name="deep_research_stylish.pdf",
                mime="application/pdf"
            )

            st.session_state.deep_running = False

        st.write_stream(stream_research)

    # === Branding at bottom ===
    st.markdown("<hr style='margin-top:50px; margin-bottom:10px;'>", unsafe_allow_html=True)
    st.markdown("🛠️ API by **WaysAhead**", unsafe_allow_html=True)

asyncio.run(main())
