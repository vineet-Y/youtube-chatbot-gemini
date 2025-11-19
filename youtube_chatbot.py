import streamlit as st
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# --- Configuration ---
LLM_MODEL = 'gemini-2.5-flash'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# --- 1. UTILITY FUNCTIONS ---

def extract_video_id(url_or_id):
    """Extracts the 11-character YouTube video ID from various URL formats."""
    # Pattern to match various YouTube URLs and extract the ID
    pattern = (
        r'(?:https?://)?(?:www\.)?'
        r'(?:youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=))|'
        r'(?:youtu\.be/|youtube\.com/shorts/))'
        r'([\w-]{11})'
    )
    match = re.search(pattern, url_or_id)
    
    if match:
        return match.group(1)
    # Check if the input is already an 11-character ID
    if len(url_or_id) == 11 and re.match(r'^[\w-]{11}$', url_or_id):
        return url_or_id
        
    return None

# --- 2. CORE RAG PIPELINE SETUP ---

# Cache the heavy resource creation (LLM, Embeddings, Vector Store)
@st.cache_resource(show_spinner="Setting up RAG and loading transcript...")
def setup_rag_pipeline(video_id: str):
    """Initializes the LLM, loads the transcript, creates vector store, and builds the LCEL chain."""

    # 1. Securely load API Key
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it securely in Streamlit Secrets.")
        st.stop()

    # 2. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1, 
        api_key=api_key
    )

    # 3. Load Transcript
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
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
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.create_documents([transcript])
    
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # --- 5. LCEL Chain Construction ---

    # A) History-Aware Retriever Prompt: Turns chat history + new question into a standalone query
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a concise, standalone search query to find relevant information from the transcript. Do not answer the question."),
    ])

    # B) History-Aware Retriever: The chain component that performs the rephrasing
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    # C) RAG Answer Prompt: The prompt that gets the final answer
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. Answer the user's question ONLY from the following video transcript context. "
         "Maintain a conversational style but strictly adhere to the provided context. "
         "If the context is insufficient, politely say you cannot answer based on the video content.\n\n"
         "Context from Transcript: {context}"),
        MessagesPlaceholder(variable_name="chat_history"), # Include chat history for continuity
        ("user", "{input}"),
    ])
    
    # D) Final Chain Composition (Contextualizing Retriever -> Answer Generator)
    final_chain = create_retrieval_chain(
        history_aware_retriever, qa_prompt | llm
    )

    return final_chain

# --- 3. STREAMLIT APPLICATION ---

def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot (Dynamic)", layout="wide")
    st.title("📹 Gemini Chatbot: Chat with Any YouTube Video")
    
    # --- Sidebar for Input ---
    with st.sidebar:
        st.header("1. Enter Video URL")
        url_input = st.text_input(
            "YouTube URL or Video ID:", 
            placeholder="e.g., https://www.youtube.com/watch?v=Gfr50f6ZBvo"
        )
        
        video_id = extract_video_id(url_input)
        
        if video_id:
            st.success(f"Video ID detected: `{video_id}`")
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.markdown("---")
            st.header("2. Start Chatting")
            # Create a unique key for the chat input to reset it when the video changes
            chat_key = f"chat_input_{video_id}"
        else:
            st.error("Please enter a valid YouTube URL or 11-character video ID.")
            return

    # --- Main App Logic ---
    
    # Setup the RAG pipeline. This will run only once per unique video ID.
    chain = setup_rag_pipeline(video_id)
    
    if chain is None:
        return # Stop if RAG setup or transcript loading failed

    # Initialize chat history in session state for the specific video
    if "video_id" not in st.session_state or st.session_state.video_id != video_id:
        st.session_state.messages = [AIMessage(content=f"Hello! I have loaded the transcript for video ID: `{video_id}`. Ask me anything about its content!")]
        st.session_state.video_id = video_id

    # Display chat messages
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    # Handle user input
    if prompt := st.chat_input("Ask a question about the video transcript...", key=chat_key):
        
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Invoke the LCEL chain using the new structure
                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.messages 
                    })
                    
                    # The response structure from create_retrieval_chain has the answer in 'answer'
                    answer = response['answer']
                    st.markdown(answer)
                    
                    # Append assistant response to chat history
                    st.session_state.messages.append(AIMessage(content=answer))
                    
                except Exception as e:
                    # Fallback if there's a serious API or chain error
                    st.error(f"An error occurred during chain execution. Please check your Gemini API Key and Streamlit Logs. Error: {e}")

if __name__ == "__main__":
    main()
