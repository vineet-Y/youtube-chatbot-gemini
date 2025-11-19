'''import streamlit as st
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# Updated LangChain imports for recent releases
from langchain.retrievers import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- 1. UTILITY FUNCTIONS ---

def extract_video_id(url_or_id: str) -> str | None:
    """Extracts the 11-character YouTube video ID from various URL formats.

    Handles full URLs, short youtu.be links, /shorts/ links or raw 11-char ids.
    """
    if not url_or_id:
        return None
    # common patterns: v=ID, /watch?v=ID, youtu.be/ID, /embed/ID, /shorts/ID
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

# --- 2. CORE RAG PIPELINE SETUP ---

# Cache the heavy resource creation (LLM, Embeddings, Vector Store)
@st.cache_resource(show_spinner="Setting up RAG and loading transcript...")
def setup_rag_pipeline(video_id: str):
    """Initializes the LLM, loads the transcript, creates vector store, and builds the retrieval chain.

    NOTE: replace FAISS/HuggingFaceEmbeddings with your preferred vectorstore/embeddings in prod.
    """

    # 1. Securely load API Key
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it securely in Streamlit Secrets.")
        st.stop()

    # 2. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        api_key=api_key,
    )

    # 3. Load Transcript
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])  # may raise
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        st.error(f"Transcripts are disabled or unavailable for video ID: `{video_id}`.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

    # 4. Chunking and Embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- 5. Chain Construction for latest LangChain ---

    # A) Rephrase prompt: rewrite follow-up Q into standalone query using chat history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Given the above conversation, generate a concise, standalone search query to find relevant information from the transcript. Do not answer the question."),
    ])

    # B) Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=rephrase_prompt
    )

    # C) QA prompt used by the document-combiner chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question ONLY from the following video transcript context. "
         "Maintain a conversational style but strictly adhere to the provided context. "
         "If the context is insufficient, politely say you cannot answer based on the video content.\n\n"
         "Context from Transcript: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    # D) Build combine_documents chain ("stuff" strategy). In newer LangChain you pass the prompt + llm here.
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    # E) Final retrieval chain ties the history-aware retriever + combiner together
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_documents_chain=combine_docs_chain,
    )

    return retrieval_chain

# --- 3. STREAMLIT APPLICATION ---


def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot (Dynamic)", layout="wide")
    st.title("📹 Gemini Chatbot: Chat with Any YouTube Video")

    # --- Sidebar for Input ---
    with st.sidebar:
        st.header("1. Enter Video URL")
        url_input = st.text_input(
            "YouTube URL or Video ID:",
            placeholder="e.g., https://www.youtube.com/watch?v=Gfr50f6ZBvo",
        )

        video_id = extract_video_id(url_input)

        if video_id:
            st.success(f"Video ID detected: `{video_id}`")
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.markdown("---")
            st.header("2. Start Chatting")
            chat_key = f"chat_input_{video_id}"
        else:
            st.error("Please enter a valid YouTube URL or 11-character video ID.")
            return

    # --- Main App Logic ---
    chain = setup_rag_pipeline(video_id)
    if chain is None:
        return

    # Initialize chat history in session state for the specific video
    if "video_id" not in st.session_state or st.session_state.video_id != video_id:
        st.session_state.messages = [
            AIMessage(content=f"Hello! I have loaded the transcript for video ID: `{video_id}`. Ask me anything about its content!")
        ]
        st.session_state.video_id = video_id

    # Display chat messages
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    # Handle user input
    if prompt := st.chat_input("Ask a question about the video transcript...", key=chat_key):
        # Add user message to chat history (LangChain-friendly form kept separately)
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Convert Streamlit message objects to simple role/content dicts for the chain
                    history_for_chain = [
                        {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
                        for m in st.session_state.messages
                    ]

                    # Invoke the retrieval chain
                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": history_for_chain,
                    })

                    # Robustly extract the answer field (different chain versions use different keys)
                    answer = response.get("answer") or response.get("output_text") or str(response)

                    st.markdown(answer)
                    st.session_state.messages.append(AIMessage(content=answer))

                except Exception as e:
                    st.error(f"An error occurred during chain execution. Please check your Gemini API Key and Streamlit Logs. Error: {e}")


if __name__ == "__main__":
    main()
'''
import streamlit as st
import langchain
import sys

def main():
    """
    A minimal Streamlit application to display the installed LangChain version.
    """
    st.set_page_config(page_title="LangChain Version Check", layout="centered")
    
    st.title("🐍 LangChain Version Status")

    try:
        # Access the __version__ attribute directly from the imported package
        version = langchain.__version__
        st.success(f"**Status: SUCCESS**")
        st.markdown(f"The installed `langchain` version is: **`{version}`**")
        st.info("This confirms that the core `langchain` package is successfully installed in your Streamlit environment.")
    except AttributeError:
        st.error("Error: Could not retrieve LangChain version.")
        st.markdown("The `langchain` package might be installed, but the `__version__` attribute is unavailable.")
    except ImportError:
        st.error("Error: The `langchain` package is NOT installed.")
        st.markdown("Please ensure `langchain` is listed in your `requirements.txt` file and your app is rebooted.")

    st.markdown("---")
    st.markdown(f"**Python Interpreter:** `{sys.version.split(' ')[0]}`")

if __name__ == "__main__":
    main()
