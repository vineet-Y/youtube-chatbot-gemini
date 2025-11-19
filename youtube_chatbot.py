# youtube_chatbot_langchain_v1_0_5.py
import streamlit as st
import os
import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_K = 4  # number of documents to retrieve

# --- Utility: extract YouTube video id ---
def extract_video_id(url_or_id: str) -> str | None:
    if not url_or_id:
        return None
    patterns = [
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"/embed/([0-9A-Za-z_-]{11})",
        r"/shorts/([0-9A-Za-z_-]{11})",
        r"/v/([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return None

# --- Helper: safely call LLM to rewrite follow-ups into standalone query ---
def rewrite_followup_to_standalone(llm: ChatGoogleGenerativeAI, chat_history: List[Dict], user_question: str) -> str:
    """
    Build a prompt that contains chat history and the follow-up question; ask LLM to return a concise standalone query.
    Returns the rewritten query string.
    """
    # turn history into a readable block
    if chat_history:
        history_text = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "User:" if role == "user" else "Assistant:"
            history_text.append(f"{prefix} {content}")
        history_block = "\n".join(history_text)
    else:
        history_block = "(no prior conversation)"

    rewrite_prompt = (
        "You are a utility that rewrites follow-up questions into concise standalone search queries. "
        "Do NOT answer — only produce the rewritten search query.\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Follow-up question: {user_question}\n\n"
        "Standalone search query:"
    )

    # ChatGoogleGenerativeAI supports .invoke(messages) where messages can be a list of tuples or a HumanMessage.
    # We use HumanMessage for compatibility and get the response content.
    response: AIMessage = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten = (response.content or "").strip()
    # defensive fallback: if LLM returns long text, try to take the first line
    if "\n" in rewritten:
        first_line = rewritten.splitlines()[0].strip()
        if first_line:
            return first_line
    return rewritten

# --- Helper: fetch relevant documents from FAISS vectorstore ---
def retrieve_documents(vector_store: FAISS, query: str, k: int = EMBEDDING_K):
    """
    Returns a list of documents (langchain Document-like objects) given the query.
    FAISS vector_store typically exposes similarity_search() or similar. We'll try both common names.
    """
    # try preferred API names in order
    if hasattr(vector_store, "similarity_search"):
        return vector_store.similarity_search(query, k=k)
    if hasattr(vector_store, "search"):
        return vector_store.search(query, k=k)
    if hasattr(vector_store, "as_retriever"):
        # try retriever path
        retr = vector_store.as_retriever(search_kwargs={"k": k})
        if hasattr(retr, "get_relevant_documents"):
            return retr.get_relevant_documents(query)
        if hasattr(retr, "retrieve"):
            return retr.retrieve(query)
    # last resort: raise
    raise RuntimeError("Unsupported vector_store API: cannot run similarity search")

# --- Helper: call LLM to answer from context ---
def answer_from_context(llm: ChatGoogleGenerativeAI, context_docs: List, user_question: str) -> str:
    """
    Build a prompt that contains the concatenated context from retrieved docs and the user's question.
    Ask LLM to answer strictly from provided context and cite when unsure.
    """
    # create a compact context string (include small separators and optional source markers)
    ctx_pieces = []
    for i, d in enumerate(context_docs, start=1):
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        # guard length (you might want to truncate very long docs)
        ctx_pieces.append(f"--- DOCUMENT {i} ---\n{text}\n")
    context_block = "\n".join(ctx_pieces).strip() or "No context extracted."

    qa_prompt = (
        "You are a helpful assistant. Answer the user's question *only* using the information in the 'Context from Transcript' below. "
        "If the context does not contain the answer, say you cannot answer from the video transcript.\n\n"
        f"Context from Transcript:\n{context_block}\n\n"
        f"User question: {user_question}\n\n"
        "Answer (be concise, and base your answer strictly on the context above):"
    )

    response: AIMessage = llm.invoke([HumanMessage(content=qa_prompt)])
    answer = (response.content or "").strip()
    return answer

# --- Setup / caching heavy resources ---
@st.cache_resource(show_spinner="Setting up RAG and loading transcript...")
def setup_rag_pipeline(video_id: str):
    # load key
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets or environment.")
        st.stop()

    # init LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, api_key=api_key)

    # load transcript
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.get("text", "") for chunk in transcript_list)
    except TranscriptsDisabled:
        st.error(f"Transcripts disabled/unavailable for video ID: {video_id}")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

    # chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.create_documents([transcript])

    # embeddings + vectorstore
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding)

    # return the LLM and vector store so caller can orchestrate manual history-aware flow
    return {"llm": llm, "vector_store": vector_store}

# --- Streamlit app ---
def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot (LangChain v1.0.5)", layout="wide")
    st.title("📹 Gemini Chatbot (LangChain v1.0.5) — Chat with a YouTube Transcript")

    with st.sidebar:
        st.header("1. Enter Video URL")
        url_input = st.text_input("YouTube URL or Video ID:", placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
        video_id = extract_video_id(url_input)

        if video_id:
            st.success(f"Video ID detected: `{video_id}`")
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.markdown("---")
            st.header("2. Start Chatting")
            chat_key = f"chat_input_{video_id}"
        else:
            st.error("Enter a valid YouTube URL or 11-character video ID.")
            return

    # setup heavy resources
    env = setup_rag_pipeline(video_id)
    if env is None:
        return
    llm = env["llm"]
    vector_store = env["vector_store"]

    # initialize session history for this video
    if "video_id" not in st.session_state or st.session_state.video_id != video_id:
        st.session_state.video_id = video_id
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I loaded the transcript for video `{video_id}`. Ask me anything about it."}
        ]

    # display messages
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # handle new user input
    if prompt := st.chat_input("Ask a question about the video transcript...", key=chat_key):
        # append user message to session history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Rewriting query, retrieving docs, and generating answer..."):
                try:
                    # 1) rewrite follow-up question to standalone (if needed)
                    rewritten_query = rewrite_followup_to_standalone(llm, st.session_state.messages[:-1], prompt)
                    # if rewriting produces an empty string, fall back to user's prompt
                    if not rewritten_query:
                        rewritten_query = prompt

                    # 2) retrieve documents from vector store
                    retrieved_docs = retrieve_documents(vector_store, rewritten_query, k=EMBEDDING_K)

                    # 3) answer using the context from retrieved docs
                    answer = answer_from_context(llm, retrieved_docs, prompt)

                    # show answer and append to history
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Chain execution failed: {e}")

if __name__ == "__main__":
    main()
