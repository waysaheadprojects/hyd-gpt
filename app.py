import os
import asyncio
import time
import threading
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st
from io import BytesIO, StringIO
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
import pandas as pd
import matplotlib.pyplot as plt
import re

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
    """Fetch latest retail news snippets about Hyderabad or Inorbit Mall as a fallback fact."""
    query = (
        "latest Hyderabad retail news OR Inorbit Mall site:business-standard.com "
        "OR site:economictimes.indiatimes.com OR site:newsmeter.in"
    )
    results = await tavily.ainvoke({"query": query})
    if results and "results" in results and results["results"]:
        items = []
        for r in results["results"]:
            title = r.get("title", "No title")
            url = r.get("url", "#")
            content = r.get("content", "")
            items.append(f"🔗 **{title}**\n{content}\n👉 [Read more]({url})\n")
        return "\n\n".join(items)
    fallback = await llm.ainvoke("Give 2 numeric facts about Hyderabad retail.")
    return fallback.content.strip()

# === Vector ===
if os.path.exists(INDEX_PATH):
    vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    st.title("🚀 Retail Hyderabad Agent")
    uploaded_file = st.file_uploader("Upload your PDF to index", type=["pdf"])
    if uploaded_file:
        with st.spinner("Indexing..."):
            os.makedirs("my-docs", exist_ok=True)
            save_path = os.path.join("my-docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
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

def get_retriever_chain():
    """Return a history-aware retriever chain for RAG."""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a precise search query.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_rag_chain(chain):
    """Return the final retrieval + answer chain for RAG with style prompt."""
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are **Retailopedia**, an intelligent analytical retail assistant.
Always provide crisp numeric answers.
If appropriate, output your data as markdown tables.
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

async def vector_lookup(query: str) -> str:
    """Try vector store lookup; fallback to LLM with a Hyderabad fact if no match."""
    docs = vs.similarity_search(query, k=5)
    if not docs:
        fallback = await llm.ainvoke(f"User said: \"{query}\". Reply politely.")
        fact = await get_latest_retail_news()
        return f"{fallback.content.strip()}\n\n💡 **Hyderabad Retail Fact:** {fact}"
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = await chain.ainvoke({"chat_history": [], "input": query, "context": context})
    return result["answer"]

class CustomLogsHandler:
    """Handle logs for GPTResearcher deep research actions."""
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    async def send_json(self, data: Dict[str, Any]) -> None:
        self.logs.append(data)

def run_gpt_researcher_sync(query: str, logs_handler: CustomLogsHandler) -> str:
    """Run GPTResearcher synchronously to return deep research report."""
    return asyncio.get_event_loop().run_until_complete(run_gpt_researcher(query, logs_handler))

vector_tool = StructuredTool.from_function(vector_lookup)

class State(BaseModel):
    """LangGraph shared state model."""
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state: State):
    """Router node: currently always route to vector search."""
    return {"route": "vector"}

async def vector_node(state: State):
    """Vector node: runs the vector lookup and returns the answer."""
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.set_entry_point("router")
graph.add_edge("router", "vector")
graph.add_edge("vector", END)
agent = graph.compile()

def plot_markdown_table(response: str):
    """If a markdown table is detected in the response, parse and plot it."""
    match = re.search(r"((\|.+\n)+)", response)
    if not match:
        return False
    table_md = match.group(1)
    lines = [line for line in table_md.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    cleaned_table = "\n".join(lines)
    try:
        df = pd.read_csv(StringIO(cleaned_table), sep="|")
        df = df.dropna(axis=1, how="all")
        df = df.drop(df.columns[0], axis=1)
    except Exception as e:
        print(f"Table parse error: {e}")
        return False

    fig, ax = plt.subplots()
    if "Year" in df.columns[0]:
        df[df.columns] = df[df.columns].apply(lambda x: pd.to_numeric(x, errors='ignore'))
        df.set_index(df.columns[0], inplace=True)
        df.plot(ax=ax, marker="o")
    else:
        df.plot(kind="bar", ax=ax)

    st.pyplot(fig)
    return True

async def main():
    """Main Streamlit app loop for Retail Agent."""
    st.markdown("<h1>Retail Agent</h1>", unsafe_allow_html=True)
    fact = await get_latest_retail_news()
    st.info(f"💡 **Retail Fact:** {fact}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
            plotted = plot_markdown_table(msg["content"])
            if not plotted:
                st.info("✅ No chart detected.")

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
        run_deep = st.button("🚀", disabled=True)  # 🚀 Deep Research button stays disabled

    if send and query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            result = await agent.ainvoke(State(query=query))
            answer = result["answer"]
            if asyncio.iscoroutine(answer):
                answer = await answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("<hr style='margin-top:50px; margin-bottom:10px;'>", unsafe_allow_html=True)
    st.markdown("🛠️ API by **WaysAhead**", unsafe_allow_html=True)

asyncio.run(main())
