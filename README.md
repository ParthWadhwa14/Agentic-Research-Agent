# Research Agentic Chatbot

Agentic AI chatbot with normal chat mode, deep research mode, file uploads, report generation, data analysis, math support, dynamic PPT generation, and MCP-ready project config.

## Stack

- Backend: FastAPI + Python + LangGraph agents
- Frontend: Next.js + React + TypeScript + TailwindCSS
- AI workflows: `research_agent.py`, `basic_agent.py`, `memory_system.py`

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn backend.main:app --reload --port 8000
```

Backend API:

- `GET /api/health`
- `POST /api/agent`
- `POST /api/agent/stream`
- `GET /api/download/{filename}`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

Optional frontend env:

```bash
cp .env.example .env.local
```

## Required AI Environment Variables

Create an uncommitted `.env` in the project root:

```bash
GROQ_API_KEY=your_key
SERPER_API_KEY=your_key
NVIDIA_API_KEY=your_key_if_using_basic_agent
GITHUB_PERSONAL_ACCESS_TOKEN=your_token_if_using_github_mcp
```

## File Inputs

The UI accepts:

- PDFs: research grounding and PDF extraction
- Images: vision/OCR context
- CSV/Excel: data analysis, statistics, charts

Output type is inferred from the prompt. Ask for a PPT, Markdown, LaTeX, code, or a plain report directly in your message.
If no report type is specified, the backend generates a `.txt` download by default.
