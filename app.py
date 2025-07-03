import os
import asyncio
import time
import threading
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from io import BytesIO

import fitz  # PyMuPDF for PDF text extraction
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langgraph.graph import StateGraph, END
from gpt_researcher import GPTResearcher
import gpt_researcher.actions.agent_creator as agent_creator

# === Setup ===
nest_asyncio.apply()
load_dotenv()
os.environ["REPORT_SOURCE"] = "local"

# === Styling ===
st.set_page_config(page_title="Perplexity-Style Research", page_icon="🧩")
st.markdown("""
    <style>
    body { background-color: white; }
    .stApp { background-color: white; }
    div[data-testid="stHeader"] { background: white; }
    </style>
""", unsafe_allow_html=True)

# === Patch GPTResearcher if needed ===
original = agent_creator.extract_json_with_regex
def safe_extract_json_with_regex(response):
    if not response:
        return None
    return original(response)
agent_creator.extract_json_with_regex = safe_extract_json_with_regex

# === LLM & Embeddings ===
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.2)
embeddings = OpenAIEmbeddings()

INDEX_PATH = "./faiss_index"

# === Load or init vector store ===
if os.path.exists(INDEX_PATH):
    vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vs = None

# === One-time PDF upload ===
if vs is None:
    st.title("📂 Upload PDF — One Time Setup")
    uploaded_file = st.file_uploader("Upload a PDF to index", type=["pdf"])
    if uploaded_file:
        with st.spinner("Indexing... Please wait."):
            pdf_bytes = uploaded_file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = [page.get_text().strip() for page in pdf_doc if page.get_text().strip()]
            docs = [{"page_content": t} for t in pages]
            if docs:
                vs = FAISS.from_documents(docs, embeddings)
                vs.save_local(INDEX_PATH)
                st.success("✅ Vector store created! Please reload the page.")
            else:
                st.warning("No text found in the PDF. Try another file.")
    st.stop()

# === Local vector + retriever chain ===
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
        ("system", """
You are a retail research assistant.
Use ONLY the provided context. If none, reply ❌.
Use markdown, bullet points, or tables if helpful.
Context: {context}
        """),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ])
    return create_retrieval_chain(chain, create_stuff_documents_chain(llm, prompt))

async def vector_lookup(query: str) -> str:
    docs = vs.similarity_search(query, k=5)
    if not docs:
        return "❌ No vector match."
    context = "\n\n".join([d.page_content for d in docs])
    chain = get_rag_chain(get_retriever_chain())
    result = await chain.ainvoke({"chat_history": [], "input": query, "context": context})
    return result["answer"]

async def chitchat_tool(query: str) -> str:
    prompt = f'User said: "{query}". Reply nicely in 1–2 short lines.'
    return (await llm.ainvoke(prompt)).content.strip()

# === Logs Handler ===
class CustomLogsHandler:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    async def send_json(self, data: Dict[str, Any]) -> None:
        self.logs.append(data)

async def run_gpt_researcher(query: str, logs_handler: CustomLogsHandler) -> str:
    researcher = GPTResearcher(
        query=query,
        report_type="research_report",
        report_source="hybrid",
        vector_store=vs,
        websocket=logs_handler
    )
    await researcher.conduct_research()
    return await researcher.write_report()

def run_gpt_researcher_sync(query: str, logs_handler: CustomLogsHandler) -> str:
    return asyncio.get_event_loop().run_until_complete(run_gpt_researcher(query, logs_handler))

async def deep_tool_fn(query: str) -> str:
    handler = CustomLogsHandler()
    return await run_gpt_researcher(query, handler)

vector_tool = StructuredTool.from_function(vector_lookup)
chitchat = StructuredTool.from_function(chitchat_tool)
deep_tool = StructuredTool.from_function(deep_tool_fn)

class State(BaseModel):
    query: str
    route: Optional[str] = None
    answer: Optional[str] = None

async def router(state: State):
    res = await llm.ainvoke(f'Classify: "{state.query}" — return vector OR chitchat.')
    return {"route": res.content.strip().lower()}

async def vector_node(state: State):
    return {"answer": await vector_tool.ainvoke({"query": state.query})}

async def chitchat_node(state: State):
    return {"answer": await chitchat.ainvoke({"query": state.query})}

async def deep_node(state: State):
    return {"answer": await deep_tool.ainvoke({"query": state.query})}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("deep", deep_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: s.route, {"vector": "vector", "chitchat": "chitchat"})
graph.add_edge("vector", END)
graph.add_edge("chitchat", END)
graph.add_edge("deep", END)

agent = graph.compile()

# === Main Streamlit UI ===
async def main():
    st.title("🧩 Perplexity-Style Research")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === Chat input ===
    prompt = st.chat_input("Ask anything...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            result = await agent.ainvoke(State(query=prompt))
            answer = result["answer"]
            if asyncio.iscoroutine(answer):
                answer = await answer

            if answer.startswith("❌"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ Nothing found locally. You can run **Deep Research 🚀** below!"
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()

    st.divider()

    # === Deep Research ===
    st.subheader("Deep Research")
    deep_query = st.text_input("Topic for Deep Research", key="deep_query")
    if st.button("🚀 Run Deep Research"):
        if not deep_query.strip():
            st.warning("Please enter a topic.")
        else:
            def stream_research():
                logs_handler = CustomLogsHandler()
                result_holder = {"report": ""}

                def run_and_store():
                    result_holder["report"] = run_gpt_researcher_sync(deep_query, logs_handler)

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
                    label="📄 Download Report as PDF",
                    data=pdf_buffer,
                    file_name="deep_research_report.pdf",
                    mime="application/pdf"
                )

            st.write_stream(stream_research)

asyncio.run(main())
