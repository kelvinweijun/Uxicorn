# Uxicorn: RAPTOR-Enhanced RAG Chatbot with Autonomous Data Analysis Agent

![Project Title](assets/title2.jpg)

A document intelligence and data analysis application combining **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval), autonomous AI agents, Excel analysis, and interactive chart generation, all powered by [Cerebras](https://cerebras.ai/) and its supported model.

---

## Features

### RAPTOR Document Intelligence
- **Hierarchical indexing**: Chunks documents into multiple levels of abstraction using K-means clustering and LLM summarization
- **Multi-level retrieval**: Fetches both high-level summaries and granular details simultaneously
- **FAISS vector index**: Fast similarity search across all tree levels
- **PDF support**: Upload and index PDF documents automatically

### Excel Data Analysis
- Upload `.xlsx` / `.xls` files with multi-sheet support
- Automatic data profiling (column types, statistics, sample rows)
- Pandas DataFrame integration for in-session analysis
- Merges Excel metadata into the RAPTOR index for unified retrieval

### Autonomous Agent Mode
An autonomous multi-step reasoning agent that can:
- **Retrieve nodes** dynamically from the RAPTOR tree
- **Execute Python code** safely on in-memory DataFrames
- **Create charts** (matplotlib static or Plotly interactive)
- **Analyze Excel data** with structured summaries and statistics
- Maintain **memory across steps** and self-direct toward a goal

### Chart Generation
- Matplotlib static charts (bar, line, scatter, pie, etc.)
- Plotly interactive charts
- Charts are automatically displayed in the chat response

### Web Search (Optional)
- Primary: [LangSearch API](https://langsearch.com/)
- Fallback: DuckDuckGo (via `duckduckgo-search` or `ddgs`)

### Qwen Chain-of-Thought Handling
- Automatically strips `<think>...</think>` tags from model output for clean responses

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web framework | Flask |
| LLM inference | Cerebras SDK (Any supported models with thinking capability) |
| Embeddings | `nomic-ai/nomic-embed-text-v1` via SentenceTransformers |
| Vector search | FAISS |
| Clustering | scikit-learn (K-means) |
| PDF parsing | PyPDF2 |
| Excel parsing | pandas + openpyxl |
| Static charts | matplotlib |
| Interactive charts | Plotly |
| Markdown rendering | `markdown` Python library |

---

## Requirements

### Python Version
Python 3.8+

### Install Dependencies

```bash
pip install flask cerebras-cloud-sdk PyPDF2 sentence-transformers faiss-cpu \
    scikit-learn numpy pandas openpyxl matplotlib plotly markdown requests \
    duckduckgo-search
```

> **Note:** For GPU-accelerated FAISS, replace `faiss-cpu` with `faiss-gpu`.

---

## API Key Setup

### Required: Cerebras API Key

Set via environment variable:
```bash
export CEREBRAS_API_KEY=your_key_here
```

Or create a `config.txt` file in the project root (one key per line, supports multiple keys for automatic rotation):
```
csk-your-first-key
csk-your-second-key
```

### Optional: LangSearch API Key (Web Search)

```bash
export LANGSEARCH_API_KEY=your_langsearch_key
```

Or create `langsearch_config.txt` with just the key:
```
lsk-your-langsearch-key
```

---

## Running the Application

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Project Structure

```
.
project/
├── app.py              # Main flask application
├── config.py           # API key management
├── models.py           # Data classes (RAPTORNode)
├── stores.py           # Global state stores
├── embeddings.py       # Embedding & RAPTOR tree logic
├── excel_utils.py      # Excel parsing & analysis
├── charts.py           # Chart generation
├── agent.py            # DataAnalysisAgent class
├── search.py           # Web search
└── utils.py            # Helpers (remove_thinking_tags, format_markdown, etc.)
├── config.txt          # Cerebras API key(s) (optional, not committed)
├── langsearch_config.txt  # LangSearch key (optional, not committed)
├── templates/
│   └── index.html      # Frontend chat UI
└── README.md
```

---

## Usage Guide

### Standard RAG Mode
1. Upload a PDF using the upload button
2. Ask questions in the chat — the system retrieves relevant RAPTOR nodes and optionally augments with web search

### Excel Data Analysis Mode
1. Upload an `.xlsx` file
2. Ask questions about the data in natural language (e.g., "What is the total revenue by region?")
3. Enable **Agent Mode** for multi-step analysis and chart generation

### Autonomous Agent Mode
Enable the **Agent** toggle in the UI. The agent will:
1. Analyze available data structure
2. Execute Python/pandas code to compute answers
3. Generate charts if helpful
4. Return a final synthesized answer

**Example prompts for agent mode:**
- "Summarize the key trends in this data and create a bar chart"
- "Calculate the average sales per month and plot a line chart"
- "Which product category has the highest total revenue?"

---

## Configuration Options (per request)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_web_search` | `true` | Augment answers with live web search |
| `use_pdf_context` | `true` | Use RAPTOR-indexed document context |
| `use_raptor` | `true` | Multi-level tree traversal (vs. leaf-only) |
| `use_agent` | `false` | Enable autonomous agent mode |
| `max_agent_steps` | `5` | Maximum reasoning steps for agent |

---

## API Key Rotation

The app supports multiple Cerebras API keys for automatic rate-limit handling. Add multiple keys to `config.txt`, one per line. When a rate limit (HTTP 429) is hit, the app automatically rotates to the next available key.

---

## Limitations & Notes

- **Code execution**: The agent executes Python code in a restricted environment. No file system access or imports are allowed inside agent code.
- **Memory**: All document and Excel data is stored **in-memory** and resets when the server restarts.
- **RAPTOR build time**: Building the tree for large PDFs may take 1–2 minutes due to LLM summarization of each cluster.
- **Model**: Uses any supported models via Cerebras. Thinking models like Qwen and GLM are preferred. Thinking tags (`<think>`) are automatically stripped from responses.

---

## License

MIT License. Use freely with attribution.
