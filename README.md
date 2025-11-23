# Multimodal RAG System

This project implements a Retrieval-Augmented Generation (RAG) pipeline capable of parsing PDFs, understanding text and charts (via OCR), and answering user queries using either local LLMs (Ollama) or hosted API models (Gemini / Groq).

## Features

- **Multimodal Ingestion:** Parses text and extracts charts/images from PDFs.
- **OCR Integration (Tesseract):** Converts charts into text descriptions for embedding.
- **Hybrid Models:** Choose local (`Mistral`, `Phi-3` via Ollama) or API (`Gemini`, `Groq`).
- **RAG Stack:** ChromaDB + Sentence-Transformers (`all-MiniLM-L6-v2`).
- **Prompt Strategies:** Zero-shot, Few-shot, Chain-of-Thought.
- **Streamlit UI:** Chat interface with retrieved source previews (text + images).

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** (Install and ensure its path is available. On Windows you may need to set `pytesseract.pytesseract.tesseract_cmd` in `ingest.py`.)
- **Ollama** (for local models):
  - Download from https://ollama.com
  - Pull models: `ollama pull mistral` and/or `ollama pull phi3`
- **API Keys (Optional):**
  - Gemini (Google Generative AI)
  - Groq
  - Store them in a `.env` file (see below). Local usage does not require keys.

## Required Libraries

The project uses the following Python libraries (see `requirements.txt`):

- **streamlit** - Web UI framework
- **langchain** - Core LangChain framework
- **langchain-community** - Community integrations for LangChain
- **langchain-huggingface** - HuggingFace embeddings integration
- **langchain-chroma** - ChromaDB vector store integration
- **langchain-google-genai** - Google Gemini API integration
- **langchain-groq** - Groq API integration
- **chromadb** - Vector database for embeddings
- **pypdf** - PDF processing utilities
- **pdfplumber** - Advanced PDF parsing (text, tables, images)
- **pytesseract** - Python wrapper for Tesseract OCR
- **Pillow** - Image processing library
- **sentence-transformers** - Embedding models and CLIP for vision
- **torch** - PyTorch for deep learning models
- **transformers** - HuggingFace transformers library
- **ollama** - Ollama API client for local models
- **python-dotenv** - Environment variable management
- **numpy** - Numerical computing library

## Installation

1. Create a virtual environment (optional):

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file (only if using API models):

```env
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
```

4. (Optional) Verify Ollama is running for local models.

## Usage

### 1. Ingest Documents

Place PDFs into `Data/`. Then build the vector store (embeddings + OCR of images):

```bash
python ingest.py
```

This creates `vector_db/` and saves cropped images to `extracted_images/`.

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

### 3. Interact

- Select a model in the sidebar (local or API).
- Choose a prompting strategy.
- Ask a question or upload an image (image text is OCR-extracted and converted into a query).
- Expand "View Retrieved Sources" under each AI answer to inspect supporting chunks.

### 4. Switching Models

- Local models require Ollama running.
- API models require keys in `.env` (loaded via `python-dotenv` in `rag_engine.py`).

### 5. Regenerating the Vector DB

If you add/remove PDFs, re-run:

```bash
python ingest.py
```

## Environment Variables

The file `.env` is ignored by Git via `.gitignore`. Example:

```env
GOOGLE_API_KEY=your_google_key_here
GROQ_API_KEY=your_groq_key_here
```

## Windows Notes

If Tesseract is installed at the default path, `ingest.py` sets:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
```

Adjust if installed elsewhere.

## Evaluation

You can inspect terminal logs for retrieval timing. For qualitative answer quality you may compare generated responses to ground truth manually or compute text overlap metrics externally (e.g., ROUGE) using saved chunks from `vector_db/`.

## Project Structure

```
app.py            # Streamlit UI
ingest.py         # PDF + image OCR ingestion, builds Chroma DB
rag_engine.py     # RAG chain: retrieval + prompting + model selection
Data/             # Place source PDFs here
vector_db/        # Persisted Chroma store (auto-generated)
extracted_images/ # Saved cropped images from PDFs
.env              # API keys (not committed)
```

## Troubleshooting

- Empty retrieval: ensure `Data/` has PDFs and ingestion ran.
- OCR weak: install Tesseract and verify path.
- Local model errors: confirm Ollama daemon running and models pulled.
- API auth errors: verify keys present in `.env` and not surrounded by quotes.

## License

Educational assignment context; add a license if distributing.
