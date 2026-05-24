# NAB Multi-Agent Banking Chatbot

A locally-hosted, multi-agent AI chatbot built for NAB (National Australia Bank) that answers banking questions across six specialist domains. The system runs entirely on your machine using [Ollama](https://ollama.com/) + Gemma 3:4b, with no cloud API keys required.

---

## Demo

![NAB Chatbot UI 1](images\photo1.png)
![NAB Chatbot UI 2](images\photo2.png)

---

## How It Works

User messages are handled by a pipeline of AI agents:

```
User Message
     │
     ▼
OrchestratorAgent  ←── Gemma 3 (via Ollama)
  • Detects intent
  • Routes to the right specialist
  • Falls back to keyword scoring if LLM skips JSON
     │
     ▼
Specialist Agent  (one of six below)
  • Retrieves relevant knowledge (FAISS semantic search or full-inject)
  • Calls Gemma 3 with KB context + conversation history
  • Returns a grounded, natural-language answer
     │
     ▼
Browser (Flask JSON response)
```

### Agent Roster

| Agent               | Domain                                           |
|---------------------|--------------------------------------------------|
| `OrchestratorAgent` | Reads intent, routes to the right specialist     |
| `AccountsAgent` | Opening/closing accounts, joint accounts, Portal Pay |
| `BusinessAgent` | Business banking, NAB Bookkeeper, EFTPOS             |
| `CardsInsuranceAgent` | Credit/debit cards, rewards, fraud, insurance  |
| `LoansAgent` | Home loans, business loans, car loans, hardship         |
| `PaymentsAgent` | Transfers, BPAY, PayID, SWIFT, FX deals              |
| `SupportAgent` | General help, passwords, branch info, escalations     |

### Knowledge Base Retrieval

- **Large KBs (> 8,000 chars)** → FAISS semantic vector search — returns the top 3 most relevant chunks for the query.
- **Small KBs (≤ 8,000 chars)** → Full-inject — the entire file is included in the prompt.
- FAISS indexes are built once at startup and cached to disk for fast subsequent restarts.

---

## Project Structure

```
multi-agent-nab-chatbot/
├── backend/
│   ├── agents.py          # All AI agent logic (orchestrator + 6 specialists)
│   └── app.py             # Flask web server & API endpoints
├── frontend/
│   └── index.html         # Single-page chat UI (vanilla HTML/CSS/JS)
├── data/
│   ├── accounts.json       # Knowledge base — accounts
│   ├── business.json       # Knowledge base — business banking
│   ├── cards-insurance.json# Knowledge base — cards & insurance
│   ├── loans.json          # Knowledge base — loans
│   ├── payments.json       # Knowledge base — payments
│   └── support.json        # Knowledge base — general support
├── cache/
│   ├── sessions.json       # Active session history (auto-cleared on restart)
│   └── faiss_indexes/      # Cached FAISS vector indexes (auto-generated)
└── requirements.txt
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10+   |
| [Ollama](https://ollama.com/download) | Latest |
| Gemma 3 model | `ollama pull gemma3:4b` |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/multi-agent-nab-chatbot.git
cd multi-agent-nab-chatbot
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Gemma 3 model via Ollama

```bash
ollama pull gemma3:4b
```

Ollama must be **running in the background** before you start the server. On macOS/Linux it starts automatically after install. On Windows, open the Ollama app first.

### 5. Start the Flask server

```bash
cd backend
python app.py
```

### 6. Open the chat UI

Navigate to [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/api/chat` | Send a message, receive an agent response |
| `DELETE` | `/api/session/<id>` | Clear conversation history for a session |
| `POST` | `/api/session/<id>/delete` | Same as DELETE (for tab-close `sendBeacon`) |
| `GET` | `/api/health` | Health check — lists all registered agents |

### Example `/api/chat` request

```json
POST /api/chat
{
  "session_id": "abc123",
  "message": "What home loans does NAB offer?"
}
```

### Example response

```json
{
  "session_id": "abc123",
  "response": "NAB offers several home loan options including...",
  "agent": "loans_agent",
  "status": "answered",
  "metrics": {
    "response_time_ms": 1842,
    "tokens_in": 512,
    "tokens_out": 148,
    "total_tokens": 660,
    "cached": false
  }
}
```

---

## Configuration

Environment variables (optional — defaults shown):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Model to use for all agents |

Set them in a `.env` file in the project root or export them before running:

```bash
export OLLAMA_MODEL=gemma3:12b   # use a larger model if your hardware supports it
```

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **AI / LLM:** Ollama, Gemma 3 (`gemma3:4b`)
- **LangChain:** `langchain`, `langchain-ollama`, `langchain-community`, `langchain-text-splitters`
- **Vector Search:** FAISS (`faiss-cpu`)
- **Frontend:** Vanilla HTML, CSS, JavaScript (no framework)

---

## Notes

- Session history is **cleared on every server restart** — this is intentional for privacy.
- FAISS indexes are persisted in `cache/faiss_indexes/` and rebuilt only when the knowledge base changes.
- The chatbot is scoped to NAB product information in the `data/` JSON files. It will not browse the live NAB website.

---

## License

This project is for educational and demonstration purposes only. It is not affiliated with or endorsed by National Australia Bank Limited.
