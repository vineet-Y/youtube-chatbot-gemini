import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Optional

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

# ---------------- CONFIG ----------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_K = 4  # documents to retrieve
TRANSCRIPT_CACHE_DIR = Path("./transcript_cache")
TRANSCRIPT_CACHE_DIR.mkdir(exist_ok=True)

# ---------------- UTILITIES ----------------

def extract_video_id(url_or_id: str) -> Optional[str]:
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


def read_uploaded_subtitles(uploaded_file: UploadedFile | None):
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        # crude cleanup for VTT/SRT
        lines = []
        for ln in content.splitlines():
            if "-->" in ln:
                continue
            if ln.strip().isdigit():
                continue
            if ln.strip().upper().startswith("WEBVTT"):
                continue
            lines.append(ln.rstrip())
        return "".join([l for l in lines if l.strip()])
    except Exception:
        return None


def download_subs_with_ytdlp(video_id: str, lang: str = "en") -> Optional[str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmpdir = tempfile.mkdtemp(prefix="ytdlp_subs_")
    out_template = os.path.join(tmpdir, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang",
        lang,
        "--output",
        out_template,
        url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
    except Exception:
        return None

    for p in Path(tmpdir).glob("*"):
        if p.suffix.lower() in (".vtt", ".srt"):
            raw = p.read_text(encoding="utf-8", errors="ignore")
            lines = []
            for ln in raw.splitlines():
                if "-->" in ln:
                    continue
                if ln.strip().isdigit():
                    continue
                if ln.strip().upper().startswith("WEBVTT"):
                    continue
                lines.append(ln.rstrip())
            text = "".join([l for l in lines if l.strip()]).strip()
            return text or None
    return None


def cache_transcript(video_id: str, transcript_text: str, meta: Dict = None) -> Path:
    path = TRANSCRIPT_CACHE_DIR / f"{video_id}.json"
    payload = {"video_id": video_id, "transcript": transcript_text, "meta": meta or {}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_cached_transcript(video_id: str) -> Optional[str]:
    path = TRANSCRIPT_CACHE_DIR / f"{video_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("transcript")
        except Exception:
            return None
    return None


# ---------------- FREE TRANSCRIPT STRATEGY ----------------

def fetch_transcript_free(video_id: str, prefer_upload: UploadedFile | None = None, lang: str = "en"):
    # 0) If user uploaded subtitles, use them
    if prefer_upload:
        txt = read_uploaded_subtitles(prefer_upload)
        if txt:
            return txt

    # 1) Try cached
    cached = load_cached_transcript(video_id)
    if cached:
        return cached

    # 2) Try youtube-transcript-api
    try:
        # ✅ correct API usage
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        text = " ".join(chunk.get("text", "") for chunk in transcript_list).strip()
        if text:
            cache_transcript(video_id, text, meta={"source": "youtube-transcript-api"})
            return text
    except Exception as e:
        # optional: log the real error instead of silently ignoring it
        print("youtube-transcript-api error:", repr(e))
        # and then we let it fall through to yt-dlp
        pass


    # 3) Fallback: yt-dlp
    subs = download_subs_with_ytdlp(video_id, lang=lang)
    if subs:
        cache_transcript(video_id, subs, meta={"source": "yt-dlp"})
        return subs

    raise RuntimeError("Could not fetch transcript. Try running locally, upload subtitle file, or ensure the video has captions.")


# ---------------- LangChain v1.0.5-compatible RAG helpers ----------------

def split_to_chunks(text: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""])
    docs = splitter.create_documents([text])
    # attach a simple source id to each chunk for citations
    for i, d in enumerate(docs):
        d.metadata = getattr(d, "metadata", {}) or {}
        d.metadata["source"] = f"chunk_{i+1}"
    return docs


def build_vector_store_from_transcript(transcript: str, embedding_model: str = EMBEDDING_MODEL):
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    chunks = split_to_chunks(transcript)
    vs = FAISS.from_documents(chunks, embedding)
    return vs


def rewrite_followup_to_standalone(llm: ChatGoogleGenerativeAI, chat_history: List[Dict], user_question: str) -> str:
    if chat_history:
        history_text = []
        for msg in chat_history:
            role = msg.get("role", "user")
            prefix = "User:" if role == "user" else "Assistant:"
            history_text.append(f"{prefix} {msg.get('content','')}")
        history_block = "".join(history_text)
    else:
        history_block = "(no prior conversation)"

    prompt = (
        "Rewrite the follow-up question into a concise standalone search query. Do NOT answer."
        f"Conversation history:{history_block}"
        f"Follow-up question: {user_question} Standalone search query:"
    )
    resp: AIMessage = llm.invoke([HumanMessage(content=prompt)])
    out = (resp.content or "").strip()
    if "" in out:
        out = out.splitlines()[0].strip()
    return out


def retrieve_docs_from_store(vector_store: FAISS, query: str, k: int = EMBEDDING_K):
    # prefer similarity_search
    if hasattr(vector_store, "similarity_search"):
        return vector_store.similarity_search(query, k=k)
    if hasattr(vector_store, "search"):
        return vector_store.search(query, k=k)
    # fallback to retriever
    retr = vector_store.as_retriever(search_kwargs={"k": k})
    if hasattr(retr, "get_relevant_documents"):
        return retr.get_relevant_documents(query)
    if hasattr(retr, "retrieve"):
        return retr.retrieve(query)
    raise RuntimeError("Vector store retrieval method not found")


def answer_with_sources(llm: ChatGoogleGenerativeAI, docs: List, user_question: str, max_context_chars: int = 8000) -> Dict:
    # build context with doc markers; truncate if too long
    pieces = []
    for i, d in enumerate(docs, start=1):
        txt = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        src = d.metadata.get("source") if getattr(d, "metadata", None) else f"doc_{i}"
        pieces.append(f"--- SOURCE: {src} ---{txt}")
    context = "".join(pieces)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided transcript context. "
        "If the answer isn't in the context, say you cannot answer from the video. Provide a short answer and then list sources (source ids) used."
        f"Context:{context}"
        f"User question: {user_question} Answer:")
    resp: AIMessage = llm.invoke([HumanMessage(content=prompt)])
    answer = (resp.content or "").strip()
    # return answer plus the doc ids so UI can show snippets
    sources = [getattr(d, "metadata", {}).get("source", f"doc_{i+1}") for i, d in enumerate(docs)]
    return {"answer": answer, "sources": sources, "docs": docs}


# ---------------- Streamlit app ----------------

@st.cache_resource(show_spinner="Setting up RAG and loading transcript...")
def setup_rag_pipeline(video_id: str, uploaded_file: UploadedFile | None, lang: str = "en"):
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets or environment.")
        st.stop()

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, api_key=api_key)

    try:
        transcript = fetch_transcript_free(video_id, prefer_upload=uploaded_file, lang=lang)
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")

    vector_store = build_vector_store_from_transcript(transcript)
    return {"llm": llm, "vector_store": vector_store, "transcript": transcript}


def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot (Free)", layout="wide")
    st.title("📹 Gemini YouTube Transcript Chatbot")

    with st.sidebar:
        st.header("Input Video or Subtitles")
        url_input = st.text_input("YouTube URL or Video ID:", placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
        uploaded_subs = st.file_uploader("Or upload subtitle file (.vtt/.srt/.txt)", type=["vtt", "srt", "txt"])
        lang = st.selectbox("Subtitle language (used for API/fallback)", options=["en", "es", "fr", "de"], index=0)
        use_cache = st.checkbox("Use cached transcript if available", value=True)
        show_sources_toggle = st.checkbox("Show source snippets with the answer", value=True)

    video_id = extract_video_id(url_input) if url_input else None

    if not video_id and not uploaded_subs:
        st.info("Enter a YouTube URL or upload subtitles to begin.")
        return

    if video_id:
        st.success(f"Video ID: {video_id}")
        st.video(f"https://www.youtube.com/watch?v={video_id}")

    # Setup RAG
    try:
        env = setup_rag_pipeline(video_id or (uploaded_subs.name if uploaded_subs else "uploaded"), uploaded_file=uploaded_subs, lang=lang)
    except Exception as e:
        st.error(str(e))
        return

    llm = env["llm"]
    vector_store = env["vector_store"]
    transcript_text = env["transcript"]

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello — I can answer questions based on the loaded video transcript/subtitles. Ask me anything!"}]

    # Chat UI
    for m in st.session_state.messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(m["content"])

    chat_key = f"chat_{video_id or 'uploaded'}"
    if prompt := st.chat_input("Ask a question about the transcript...", key=chat_key):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... retrieving relevant context..."):
                try:
                    history = st.session_state.messages[:-1]
                    rewritten = rewrite_followup_to_standalone(llm, history, prompt)
                    if not rewritten:
                        rewritten = prompt

                    docs = retrieve_docs_from_store(vector_store, rewritten, k=EMBEDDING_K)
                    res = answer_with_sources(llm, docs, prompt)
                    answer = res.get("answer")

                    # show answer
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # optionally show sources/snippets
                    if show_sources_toggle:
                        st.markdown("**Sources used:**")
                        for i, d in enumerate(res.get("docs", []), start=1):
                            src = d.metadata.get("source", f"doc_{i}") if getattr(d, "metadata", None) else f"doc_{i}"
                            snippet = (getattr(d, "page_content", None) or getattr(d, "content", None) or str(d))[:400]
                            st.markdown(f"- `{src}` — {snippet}...")

                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")


if __name__ == "__main__":
    main()
