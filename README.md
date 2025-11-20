# 🎥 YouTube Transcript Chatbot (Gemini + LangChain v1.0.5)

A powerful chatbot that answers user queries **based on the content of any YouTube video**. It extracts the transcript (using free methods), builds a retrieval-augmented generation (RAG) pipeline, and allows rich conversational querying with context awareness.

This project is designed to work even when YouTube blocks transcript requests from cloud IPs by using fallback options like **yt-dlp**, and also supports **manual subtitle uploads**.

The app can be accessed at https://youtube-chatbot-gemini-bwskskjvjpmpyjnd5qpazr.streamlit.app/
---

#  Features

###  Free-first transcript extraction

### Free-first transcript extraction

- Primary: `youtube-transcript-api` (free)
- Fallback: `yt-dlp` auto-generated subtitles (free), when environment allows
- Manual:
  - Upload `.vtt`, `.srt`, or `.txt` subtitle files
  - (Optional) Extend to support pasting raw transcript text
- Caching system to avoid re-fetching transcripts for the same video

### RAG-based Conversational QA

- Uses LangChain 1.0.5 compatible pipeline
- LLM-based, history-aware question rewriting
- FAISS vector store for fast semantic retrieval
- Gemini 2.5 Flash for rewriting and answering
- Source snippet display for transparency

### Multi-video session memory

- Every video loaded in the current Streamlit session gets:
  - Its own transcript
  - Its own FAISS vector store
  - Its own metadata (`video_key`, `video_index`, `video_is_newest`)
- All videos are stored in `st.session_state`, so the chatbot can:
  - Remember multiple videos at once
  - Know which video is the newest
  - Answer questions that reference the “new video”, “previous video” or “first video”
- Retrieval can aggregate chunks from **all** loaded videos, so you can ask:
  - “Compare the latest video with the first one.”
  - “How does the explanation of eigenvalues differ between the last two videos?”
  - “Which video gives a more intuitive explanation of X?”

### Streamlit UI

- Enter a YouTube URL or 11-character video ID
- Or upload subtitles manually
- Interactive chat interface using `st.chat_message`
- Toggle to show “sources used”
- Language selection for transcript extraction
- Session-level memory across multiple videos

---
#  Project Structure

```
youtube-chatbot-gemini/
│
├── youtube_chatbot.py        # Main Streamlit application
├── README.md                 # This file
└── requirements.txt          # Dependencies

```

---

#  How It Works (Architecture)

## 1. Transcript Retrieval

The system attempts transcript extraction in the following order:

1. **Uploaded subtitles** (if provided)
2. **Cached transcript** (if previously fetched)
3. **YouTube Transcript API**
4. **yt-dlp auto subtitles** (fallback, when environment + JS runtime allow it)

This ensures maximum reliability without using paid proxy services.

---

## 2️. Chunking & Vector Store

* Transcript is split into overlapping chunks with `RecursiveCharacterTextSplitter`
* Each chunk is embedded using **HuggingFace MiniLM** embeddings
* Stored in **FAISS** vector database
* Each chunk is given metadata such as:

source: simple chunk ID like chunk_1, chunk_2, etc.

In multi-video mode, additional metadata is attached at retrieval time:

video_key (e.g. YouTube ID or an uploaded::filename key)

video_index (0 = oldest, higher = newer)

video_is_newest (boolean flag)

---

## 3️. History-Aware Conversational RAG

### Step A: Rewrite Query

The model rewrites the user’s follow-up question into a standalone search query, using the full conversation history stored in st.session_state.messages.

Example:

User: “What happened after that?”

LLM rewrites to:

“Explain the turning point discussed in the video after the speaker mentions project failures.”

This rewritten query is used to retrieve the most relevant chunks from the vector stores.

---

### Step B: Retrieve Relevant Chunks

The app queries FAISS with the rewritten query.

In multi-video mode:

It iterates over all known videos in the session (stored in st.session_state.video_envs and st.session_state.video_order).

For each video’s vector store, it retrieves top-k chunks (e.g. k = EMBEDDING_K // 2).

It annotates each retrieved chunk with video metadata (video_key, video_index, video_is_newest).

It combines all retrieved chunks into a single context for the LLM.

This allows the model to see, in the prompt, which chunks come from which video and which video is the “newest”.

---

### Step C: Answer From Context Only

Gemini is instructed to answer strictly from the provided context:

*If the answer is not present in the transcript context, the bot explicitly says that it cannot answer from the videos.

*The final response can naturally differentiate between videos using the metadata (e.g. “In the first video…” vs “In the latest video…”).

*The UI displays source snippets for transparency.

---

## 4. Multi-Video Memory & Cross-Video Reasoning

Within a single Streamlit session:

Each time you load a new video (new URL or new uploaded subtitle file):

The app builds a new transcript + FAISS vector store via setup_rag_pipeline.

It stores this environment in st.session_state.video_envs[video_key].

It appends the key to st.session_state.video_order.

The most recently loaded video is treated as the “newest”.

At query time:

The app:

Rewrites the user’s query.

Retrieves relevant chunks from all videos’ vector stores.

Attaches video_key, video_index, and video_is_newest metadata to each chunk.

The LLM sees all this in the context, so questions like:

“Compare the new video and the previous video”

“How is the explanation in the first video different from the last one?” can be answered without custom hard-coded branching: the model uses the metadata to know what “newest” and “previous” refer to.

---

#  Installation & Setup

### 1️. Clone the repo

```bash
git clone https://github.com/vineet-Y/youtube-chatbot-gemini
cd youtube-chatbot-gemini
```

### 2️. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended packages:

```
streamlit
youtube-transcript-api
yt-dlp
langchain
langchain-core
langchain-community
langchain-google-genai
sentence-transformers
faiss-cpu
```

### 3️. Add your Gemini API key

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

### 4️. Run the app

```bash
streamlit run youtube_chatbot.py
```

---

#  Usage

1. Enter a YouTube URL or 11-character video ID
2. (Optional) Upload subtitles if YouTube transcript is unavailable
3. Ask any question about the video
4. Toggle **Show sources** to see exactly which parts of the transcript were used

---

#  Limitations

* Videos without captions (auto or manual) cannot be processed unless subtitles are uploaded
* yt-dlp fallback requires the yt-dlp binary to be installed and available in PATH
* Cloud environments (AWS/GCP/Streamlit Cloud) may have YouTube transcript API blocked — fallback helps but not guaranteed

---

#  Future Improvements

* Better subtitle parsing using an SRT/VTT parser
* Support for multilingual RAG
* Support for Whisper transcription for videos without captions (requires compute)
* Option to display timestamps for each retrieved chunk




---

Enjoy exploring YouTube videos with AI! 🎬🤖
