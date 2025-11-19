import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.prompts import PromptTemplate

# --- Configuration ---
# The video ID provided by the user (you can make this an input field later if you want)
VIDEO_ID = "Gfr50f6ZBvo"
# The model used for RAG and chat
LLM_MODEL = 'gemini-2.5-flash'

# --- 1. Data Loading and Setup Functions ---

@st.cache_resource
def load_and_process_transcript(video_id):
    """Loads the transcript, splits it, and creates the vector store."""
    try:
        # Fetch the English transcript
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        
        # Flatten the transcript to plain text
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
        
        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create documents/chunks
        chunks = splitter.create_documents([transcript])
        
        # Initialize the embedding model (cached for efficiency)
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Create the FAISS vector store
        vector_store = FAISS.from_documents(chunks, embedding)
        
        st.toast("Transcript loaded and vector store created successfully!", icon="✅")
        return vector_store, transcript
        
    except TranscriptsDisabled:
        # Note: If this fails, the app still runs but the transcript functions fail.
        st.error(f"Captions are disabled or unavailable for video ID: {video_id}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during transcript processing: {e}")
        return None, None

def initialize_chat_chain(vector_store, api_key):
    """Initializes the Conversational Retrieval Chain with memory."""
    
    # 1. Initialize the Chat Model
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1, 
        api_key=api_key
    )

    # 2. Define the Custom Prompt Template
    custom_template = """You are a friendly and helpful assistant specializing in answering questions about a YouTube video transcript.
    
    You must answer ONLY based on the following context. If the context is insufficient, politely say that you don't have enough information from the video transcript.
    
    Chat History:
    {chat_history}
    
    Context from Transcript:
    {context}
    
    Question: {question}
    """
    CUSTOM_PROMPT = PromptTemplate.from_template(custom_template)

    # 3. Initialize Memory (using a session-specific buffer)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # 4. Initialize the Conversational Retrieval Chain (RAG + Memory)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        verbose=False # Set to True for debugging
    )
    
    return chain

# --- 2. Streamlit Application Layout and Logic ---

def main():
    st.set_page_config(page_title="YouTube Transcript Chatbot", layout="centered")
    
    # Title and Configuration
    st.title("🤖 Transcript RAG Chatbot")
    st.markdown(f"**Video ID:** `{VIDEO_ID}`")
    st.markdown("Ask me anything about the content of this video!")
    
    # --- API KEY HANDLING (Using Streamlit Secrets) ---
    # We check the st.secrets dictionary for the key
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("Gemini API Key not found in Streamlit Secrets. Please configure it in your deployment settings.")
        return

    # 1. Load Data and Setup
    vector_store, transcript = load_and_process_transcript(VIDEO_ID)
    
    if not vector_store:
        return

    # 2. Initialize Chat Chain 
    if "chat_chain" not in st.session_state or st.session_state.chat_chain is None:
        try:
            st.session_state.chat_chain = initialize_chat_chain(vector_store, api_key)
            # Initialize chat history for the UI
            if "messages" not in st.session_state:
                st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize the LLM model. Check the API Key in secrets. Error: {e}")
            st.session_state.chat_chain = None
            return

    # 3. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Handle User Input
    if prompt := st.chat_input("What is the main topic of the video?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # invoke the ConversationalRetrievalChain
                    response = st.session_state.chat_chain.invoke({"question": prompt})
                    
                    # The response object from ConversationalRetrievalChain is a dict with 'answer'
                    assistant_response = response['answer']
                    
                    st.markdown(assistant_response)
                
                except Exception as e:
                    assistant_response = f"An error occurred while generating the response: {e}"
                    st.error(assistant_response)

        # Add assistant response to session history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # The chain's internal memory is automatically updated with each successful run

if __name__ == "__main__":
    main()