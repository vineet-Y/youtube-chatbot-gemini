# youtube_chatbot_free_fallback.py
import streamlit as st
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# --- Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_K = 4  # number of docs to fetch

# ------------------- FREE TRANSCRIPT HELPERS -------------------

def download_subs_with_ytdlp(video_id: str, lang: str = "en") -> str | None:
    """
    Use yt-dlp to download (auto) subtitles for the video and convert to plain text.
    Returns None if subtitles are not available or yt-dlp not found.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmpdir = tempfile.mkdtemp(prefix="ytdlp_subs_")
    out_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-sub",  # try auto-generated subtitles
        "--sub-lang", lang,
        "--output", out_template,
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
    except FileNotFoundError:
        # yt-dlp binary not available
        return None
    except subprocess.CalledProcessError:
        return None
    except subprocess.TimeoutExpired:
        return None

    # find a .vtt or .srt file and convert to text
    for p in Path(tmpdir).glob("*"):
        if p.suffix.lower() in (".vtt", ".srt"):
            raw = p.read_text(encoding="utf-8", errors="ignore")
            lines = []
            for ln in raw.splitlines():
                # drop timestamps and indices and headers
                if "-->" in ln:
                    continue
                if ln.strip().isdigit():
                    continue
                if ln.strip().upper().startswith("WEBVTT"):
                    continue
                lines.append(ln.rstrip())
            text = "\n".join([l for l in lines if l.strip()]).strip()
            # cleanup (best effort)
            try:
                for f in Path(tmpdir).glob("*"):
                    f.unlink()
                Path(tmpdir).rmdir()
            except Exception:
                pass
            return text or None
    return None


def fetch_transcript_free(video_id: str) -> str:
    """
    Free-first strategy:
    1) Try youtube-transcript-api
    2) Fallback to yt-dlp
    Raises RuntimeError if neither works.
    """
    # 1) try youtube_transcript_api
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        return " ".join(chunk.get("text", "") for chunk in transcript_list).strip()
    except TranscriptsDisabled:
        # transcript disabled according to the library — fall back to yt-dlp
        pass
    except NoTranscriptFound:
        pass
    except Exception:
        # generic failure (often IP blocked) — fall back to yt-dlp
        pass

    # 2) fallback: yt-dlp (free)
    subs = download_subs_with_ytdlp(video_id, lang="en")
    if subs:
        return subs

    # 3) nothing worked
    raise RuntimeError(
        f"Could not fetch transcript for video id {video_id}. "
        "Tried youtube-transcript-api and yt-dlp. "
        "If you run this on a cloud VM that is blocked, try running locally (home IP) "
        "or check if the video actually has captions."
    )

# ------------------- LangChain v1.0.5 compatible RAG flow -------------------

def extract_video_id(url_or_id: str) -> str | None:
    """Extract 11-char youtube id from URL or raw id"""
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


def rewrite_followup_to_standalone(llm: ChatGoogleGenerativeAI, chat_history: List[Dict], user_question: str) -> str:
    """
    Ask the LLM to rewrite a follow-up question into a standalone query given chat_history.
    """
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
    response: AIMessage = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten = (response.content or "").strip()
    if "\n" in rewritten:
        first_line = rewritten.splitlines()[0].strip()
        if first_line:
            return first_line
    return rewritten


def retrieve_documents(vector_store: FAISS, query: str, k: int = EMBEDDING_K):
    """
    Get relevant documents from FAISS-compatible vector store.
    Supports common wrapper method names.
    """
    if hasattr(vector_store, "similarity_search"):
        return vector_store.similarity_search(query, k=k)
    if hasattr(vector_store, "search"):
        return vector_store.search(query, k=k)
    if hasattr(vector_store, "as_retriever"):
        retr = vector_store.as_retriever(search_kwargs={"k": k})
        if hasattr(retr, "get_relevant_documents"):
            return retr.get_relevant_documents(query)
        if hasattr(retr, "retrieve"):
            return retr.retrieve(query)
    raise RuntimeError("Unsupported vector_store API: cannot run similarity search")


def answer_from_context(llm: ChatGoogleGenerativeAI, context_docs: List, user_question: str) -> str:
    """
    Ask LLM to answer strictly using the provided context_docs. Return answer text.
    """
    ctx_pieces = []
    for i, d in enumerate(context_docs, start=1):
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        ctx_pieces.append(f"--- DOCUMENT {i} ---\n{text}\n")
    context_block = "\n".join(ctx_pieces).strip() or "No context extracted."

    qa_prompt = (
        "You are a helpful assistant. Answer the user's question *only* using the information in the 'Context from Transcript' below. "
        "If the context does not contain the answer, say you cannot answer from the video transcript.\n\n"
        f"Context from Transcript:\n{context_block}\n\n"
        f"User question: {user_question}\n\n"
        "Answer (be concise and base your answer strictly on the context above):"
    )

    response: AIMessage = llm.invoke([HumanMessage(content=qa_prompt)])
    return (response.content or "").strip()

# ------------------- Setup / caching -------------------

@st.cache_resource(show_spinner="Setting up RAG and loading transcript...")
def setup_rag_pipeline(video_id: str):
    # load key (streamlit secrets or env)
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets or environment.")
        st.stop()

    # init LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, api_key=api_key)

    # load transcript using free-first approach
    try:
        transcript = fetch_transcript_free(video_id)
    except Exception as e:
        # bubble the error message so UI can show it
        raise RuntimeError(f"Error fetching transcript: {e}")

    # chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.create_documents([transcript])

    # embeddings + FAISS vector store
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding)

    return {"llm": llm, "vector_store": vector_store}


# ------------------- Streamlit app -------------------

def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot (Free Fallback)", layout="wide")
    st.title("📹 Gemini Chatbot — Free transcript fallback (youtube-transcript-api → yt-dlp)")

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
    try:
        env = setup_rag_pipeline(video_id)
    except Exception as e:
        st.error(str(e))
        return

    llm = env["llm"]
    vector_store = env["vector_store"]

    # session state for this video
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

    # handle user input
    if prompt := st.chat_input("Ask a question about the video transcript...", key=chat_key):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Rewriting query, retrieving docs, and generating answer..."):
                try:
                    # rewrite follow-up to standalone query using history (excluding current user message)
                    history_for_rewrite = st.session_state.messages[:-1]
                    rewritten_query = rewrite_followup_to_standalone(llm, history_for_rewrite, prompt)
                    if not rewritten_query:
                        rewritten_query = prompt

                    # retrieve docs
                    retrieved_docs = retrieve_documents(vector_store, rewritten_query, k=EMBEDDING_K)

                    # answer from context
                    answer = answer_from_context(llm, retrieved_docs, prompt)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Chain execution failed: {e}")


if __name__ == "__main__":
    main()
