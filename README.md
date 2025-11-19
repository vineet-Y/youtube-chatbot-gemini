# 🎥 YouTube Transcript Chatbot (Gemini + LangChain v1.0.5)

A powerful chatbot that answers user queries **based on the content of any YouTube video**. It extracts the transcript (using free methods), builds a retrieval-augmented generation (RAG) pipeline, and allows rich conversational querying with context awareness.

This project is designed to work even when YouTube blocks transcript requests from cloud IPs by using fallback options like **yt-dlp**, and also supports **manual subtitle uploads**.

---

#  Features

###  Free-first transcript extraction

* Primary: `youtube-transcript-api` (free)
* Fallback: `yt-dlp` auto-generated subtitles (free)
* Manual: User can upload `.vtt`, `.srt`, or `.txt` subtitles
* Caching system to avoid re-fetching transcripts

###  RAG-based Conversational QA

* Uses **LangChain 1.0.5** compatible pipeline
* Manual **history-aware question rewriting** (LLM-based)
* FAISS vector store for fast semantic retrieval
* Gemini 2.5 Flash for rewriting + answering
* Source snippet display for transparency

###  Streamlit UI

* Enter video URL or upload subtitles
* Interactive chat interface
* Toggle to show “sources used”
* Language selection for transcript

---

#  Project Structure

```
youtube-chatbot-gemini/
│
├── youtube_chatbot.py        # Main Streamlit application
├── transcript_cache/         # Cached transcripts (auto-created)
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
4. **yt-dlp auto subtitles** (fallback)

This ensures maximum reliability without using paid proxy services.

---

## 2️. Chunking & Vector Store

* Transcript is split into overlapping chunks with `RecursiveCharacterTextSplitter`
* Each chunk is embedded using **HuggingFace MiniLM** embeddings
* Stored in **FAISS** vector database
* Chunks receive metadata labels like `chunk_1`, `chunk_2` for source tracking

---

## 3️. History-Aware Conversational RAG

### Step A: Rewrite Query

The model rewrites the user's follow-up question into a **standalone query** using conversation history.

Example:

> User: "What happened after that?"

LLM rewrites to:

> "Explain the turning point discussed in the video after the speaker mentions project failures."

---

### Step B: Retrieve Relevant Chunks

FAISS retrieves the top-k chunks based on the rewritten query.

---

### Step C: Answer From Context Only

Gemini is instructed to answer **strictly from provided context**.

* If the answer is not in the transcript, the bot explicitly states it.
* Shows used sources (video chunk IDs).

---

# 🛠️ Installation & Setup

### 1️. Clone the repo

```bash
git clone https://github.com/your-username/youtube-chatbot-gemini.git
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

# 📌 Usage

1. Enter a YouTube URL or 11-character video ID
2. (Optional) Upload subtitles if YouTube transcript is unavailable
3. Ask any question about the video
4. Toggle **Show sources** to see exactly which parts of the transcript were used

---

# 📉 Limitations

* Videos without captions (auto or manual) cannot be processed unless subtitles are uploaded
* yt-dlp fallback requires the yt-dlp binary to be installed and available in PATH
* Cloud environments (AWS/GCP/Streamlit Cloud) may have YouTube transcript API blocked — fallback helps but not guaranteed

---

# ✨ Future Improvements

* Better subtitle parsing using an SRT/VTT parser
* Support for multilingual RAG
* Support for Whisper transcription for videos without captions (requires compute)
* Option to display timestamps for each retrieved chunk

---

# 🤝 Contributing

Pull requests are welcome! If you find issues or have feature ideas, feel free to open an issue.

---

# 📜 License

This project is under the MIT License.

---

# 💬 Contact

For questions or improvements, open an issue on GitHub or reach out at:
**[your-email@example.com](mailto:your-email@example.com)**

---

Enjoy exploring YouTube videos with AI! 🎬🤖
