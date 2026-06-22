# LLM Stack

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

A curated collection of **LLM application prototypes** built while exploring RAG, prompt engineering, and conversational AI patterns. Each subfolder is a standalone mini-project you can run independently.

## Projects

| Folder | Description | Key tech |
|--------|-------------|----------|
| [`Q&A Chatbot USing LLM`](Q&A%20Chatbot%20USing%20LLM/) | Streamlit Q&A chatbot over documents | LangChain, OpenAI |
| [`chatmultipledocuments`](chatmultipledocuments/) | Chat with multiple PDFs | LangChain, embeddings |
| [`pdf_query`](pdf_query/) | PDF question-answering notebook | LangChain, Jupyter |
| [`Prompt-Engineering-LangChain`](Prompt-Engineering-LangChain/) | Prompt engineering patterns | LangChain, Jupyter |
| [`Text summarization`](Text%20summarization/) | Text summarization pipeline | OpenAI, LangChain |
| [`celebrity_search_application`](celebrity_search_application/) | Semantic celebrity image search | Embeddings, Python |
| [`Image_Retrieval_System`](Image_Retrieval_System/) | Image retrieval with vector search | CV + embeddings |
| [`Conversational Q&A Chatbot`](Conversational%20Q&A%20Chatbot/) | Multi-turn conversational bot | LLM, Python |
| [`Blog Generation`](Blog%20Generation/) | AI blog post generation | LLM prompting |
| [`LLM Generic APP`](LLM%20Generic%20APP/) | Generic LLM app template | Python, Jupyter |

## Quick start

Each subproject has its own dependencies. Typical setup:

```bash
cd "Q&A Chatbot USing LLM"
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env
streamlit run app.py
```

## Why this repo

Useful as a **reference library** for common LLM patterns: document Q&A, summarization, prompt engineering, and retrieval — before graduating to production stacks like PydanticAI + MCP.

## Related

- [Robinhood AI Portfolio Copilot](https://github.com/BPrakhar30/Robinhood_AI_Portfolio_Analyzer) — production agent architecture
- [YouTube Video Summarizer](https://github.com/BPrakhar30/youtube-video-summarizer) — video summarization app
