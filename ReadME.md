# Agentic AI - Multi-Agent Framework

A comprehensive AI agentic framework combining financial market analysis, chatbot capabilities, and Model Context Protocol (MCP) servers for intelligent agent-based automation.

## 📋 Project Overview

This project implements multiple specialized AI agents:

1. **Stock Agent** - Financial market analysis and stock data retrieval
2. **Chat Bot** - Conversational AI with RAG capabilities and real-time data integration
3. **MCP Servers** - Model Context Protocol servers for agent inter-communication

### Core Features

- **Stock Data Analysis**: Integration with Alpha Vantage, Finnhub, and Yahoo Finance APIs
- **RAG (Retrieval-Augmented Generation)**: Chat with custom documents and web content
- **Sentiment Analysis**: Real-time sentiment analysis using TextBlob
- **MCP Protocol**: FastMCP-based server architecture for agent communication
- **Gradio Interface**: User-friendly web interfaces for interactions
- **LangChain Integration**: Advanced prompt chaining and agent workflows

## 🏗️ Project Architecture

### System Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENTIC AI FRAMEWORK                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
         │ STOCK AGENT  │ │  CHAT BOT   │ │ MCP SERVERS  │
         └──────────────┘ └─────────────┘ └──────────────┘
                │             │                  │
        ┌───────┼───────┐     │          ┌───────┼───────┐
        │       │       │     │          │       │       │
        ▼       ▼       ▼     ▼          ▼       ▼       ▼
    ┌───────┐┌─────┐┌──────┐┌────────┐┌─────┐┌────┐┌─────┐
    │Alpha  ││Finn │││Yahoo ││LangChain││GPT  ││RAG ││Grad │
    │Vantage││hub  ││Finance││Community││-4   ││    ││io   │
    └───────┘└─────┘└──────┘└────────┘└─────┘└────┘└─────┘
        │       │       │        │         │       │      │
        └───────┴───────┴────────┴─────────┴───────┴──────┘
                        │
                        ▼
            ┌──────────────────────────────┐
            │  DATA PROCESSING & ANALYSIS  │
            │  - Stock History Processing  │
            │  - Trade History Extraction  │
            │  - Sentiment Analysis        │
            └──────────────────────────────┘
```

### Data Flow Diagram

```
User Input
    │
    ├─► Stock Agent (ticker-based queries)
    │       │
    │       ├─► Alpha Vantage API
    │       ├─► Finnhub API  
    │       └─► Yahoo Finance API
    │
    ├─► Chat Bot (conversational queries)
    │       │
    │       ├─► RAG System (Document Search)
    │       ├─► Web Search (Tavily)
    │       └─► LLM Processing (OpenAI)
    │
    └─► MCP Servers (inter-agent communication)
            │
            ├─► Sentiment Analysis
            └─► Data Aggregation
    │
    ▼
Response to User
```

## 📁 Project Structure

```
agentic_ai/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── README.md              # This file
│
├── stock_agent/           # Stock market analysis agent
│   ├── alpha_stock_data.py        # Alpha Vantage API integration
│   ├── stock_history_utils.py     # Stock data processing utilities
│   ├── mcp_server.py              # MCP server for stock data
│   ├── mcp_client.py              # MCP client for stock queries
│   └── __pycache__/
│
├── chat_bot/              # Conversational AI with RAG
│   ├── chatbot.ipynb              # Main chatbot notebook
│   ├── RAG.ipynb                  # RAG implementation
│   ├── langchain.ipynb            # LangChain workflows
│   ├── yahoo.ipynb                # Yahoo Finance integration
│   ├── youtube_transcript.ipynb    # YouTube transcript processing
│   ├── dates_prices.json          # Sample market data
│   └── trade_history.json         # Historical trade data
│
├── mcp/                   # Generic MCP utilities
│   ├── mcp_server.py              # Sentiment analysis MCP server
│   └── mcp_client.py              # MCP client utilities
│
└── .venv/                 # Virtual environment
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.13+
- Virtual environment (venv or conda)
- API Keys for:
  - OpenAI (GPT-4)
  - Alpha Vantage
  - Finnhub
  - Tavily (web search)

### Step 1: Clone and Setup Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd agentic_ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e .
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root directory with the following API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
ALPHA_API_KEY=your_alpha_vantage_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Note**: Never commit `.env` file to version control!

### Step 4: Running the Application

#### Option A: Run Main Application
```bash
python main.py
```

#### Option B: Run Stock Agent MCP Server
```bash
cd stock_agent
python mcp_server.py
```

#### Option C: Run Chat Bot with Notebooks
```bash
cd chat_bot
jupyter notebook
# Then open chatbot.ipynb or RAG.ipynb
```

#### Option D: Run Sentiment Analysis Server
```bash
cd mcp
python mcp_server.py
# Access Gradio interface at http://localhost:7860
```

## 📦 Key Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `langchain` | LLM orchestration and chains | Latest |
| `langchain-openai` | OpenAI integration | Latest |
| `langchain-community` | Community tools and utilities | Latest |
| `langchain-chroma` | Vector store for embeddings | Latest |
| `mcp` | Model Context Protocol | 1.10.1 |
| `fastapi` | MCP server framework | 0.116.1 |
| `gradio` | Web UI for agents | 5.36.2 |
| `requests` | HTTP client for APIs | Latest |
| `beautifulsoup4` | Web scraping | Latest |
| `yfinance` | Yahoo Finance data | Latest |
| `finnhub` | Finnhub stock data | Latest |
| `tavily-python` | Web search integration | Latest |
| `python-dotenv` | Environment variable management | Latest |

## 🚀 Usage Examples

### Stock Analysis Agent

```python
from stock_agent.alpha_stock_data import get_alpha_stock_data, extract_trade_history

# Fetch stock data for Apple
data = get_alpha_stock_data("AAPL")

# Extract trade history
trade_history = extract_trade_history(data)
print(trade_history)
```

### Interactive Chat with RAG

```bash
cd chat_bot
jupyter notebook chatbot.ipynb
# Then interact with the chatbot interface
```

### Sentiment Analysis API

```bash
# Terminal 1: Start the server
cd mcp
python mcp_server.py

# Terminal 2: Make requests (automatic Gradio interface at http://localhost:7860)
# Or use the client:
cd mcp
python mcp_client.py
```

## 🔌 MCP Server Architecture

The project uses **Model Context Protocol (MCP)** for secure agent-to-agent communication:

### Available MCP Tools

#### Stock Agent MCP Server
- Analyze stock ticker
- Get historical data
- Extract trade history
- Calculate technical indicators

#### Sentiment Analysis MCP Server
- Text sentiment analysis
- Polarity scoring
- Subjectivity analysis

### Running MCP Servers

Each MCP server exposes REST APIs:

```bash
# Stock Agent (default: localhost:8000)
cd stock_agent
python mcp_server.py

# Sentiment Analysis (default: localhost:7860 with Gradio UI)
cd mcp
python mcp_server.py
```

## 📊 Integrated APIs

| API | Purpose | Usage |
|-----|---------|-------|
| **Alpha Vantage** | Time series stock data | Daily prices, technical indicators |
| **Finnhub** | Real-time market data | Live quotes, company news |
| **Yahoo Finance** | Historical stock prices | Long-term historical data |
| **OpenAI (GPT-4)** | Language model | Analysis and report generation |
| **Tavily** | Web search | Research and information gathering |
| **LangChain Hub** | Pre-built prompts | Prompt templates and chains |

## 📝 Development Notes

- **Jupyter Notebooks**: All notebooks support interactive development
- **MCP Servers**: Built on FastAPI for production-grade reliability
- **Credentials**: Store all sensitive keys in `.env` (never in code)
- **Data Files**: `dates_prices.json` and `trade_history.json` contain sample data
- **Python Version**: Requires Python 3.13+

## 🐛 Troubleshooting

### Issue: "API key not found"
```bash
# Solution: Verify .env file exists in project root
# Check file contains: OPENAI_API_KEY, ALPHA_API_KEY, etc.
cat .env
```

### Issue: "ModuleNotFoundError"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Port 7860 already in use"
```bash
# Solution: Either stop existing process or change port
# In mcp_server.py, modify: server_port=7861
```

### Issue: "Virtual environment not activated"
```bash
# Solution: Activate virtual environment
# Windows: .venv\Scripts\activate.bat
# macOS/Linux: source .venv/bin/activate
```

## 📄 License

Agentic AI Framework - 2026

## 🤝 Contributing

To add new agents or features:

1. Create new folder in project root (e.g., `new_agent/`)
2. Implement MCP server/client files if needed
3. Add configuration to `.env`
4. Update this README with new components
5. Test all integrations with other agents

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Gradio Documentation](https://www.gradio.app/docs/)

---

**Last Updated**: January 25, 2026  
**Python Version**: 3.13+  
**Framework**: LangChain + MCP
