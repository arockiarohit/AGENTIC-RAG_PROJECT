# District RAG Chatbot 🏛️

An intelligent, production-grade Agentic RAG chatbot with live Google Search Grounding (Gemini), location news aggregation, photorealistic image generation, and hybrid FAISS + BM25 document search.

---

## Project Structure

```text
district-rag-chatbot/
├── backend/                  # FastAPI Backend API & LangGraph Agents
│   ├── app/
│   │   ├── graph/            # LangGraph workflow, routers, & nodes
│   │   ├── services/         # Gemini Search, Groq LLM, Retriever, News
│   │   ├── tools/            # Web Search, Image Gen, Safe Calculator
│   │   ├── config.py         # App configuration & environment settings
│   │   ├── main.py           # FastAPI entrypoint
│   │   └── vector_store.py   # FAISS Vector Store manager
│   ├── data/                 # District records & datasets
│   ├── faiss_index/          # Pre-built FAISS vector index & metadata
│   ├── tests/                # Verification & test scripts
│   ├── Dockerfile            # Container definition for backend
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React (Vite + Tailwind CSS) Frontend
│   ├── src/                  # React components & UI logic
│   ├── public/               # Public assets
│   ├── package.json          # Node dependencies & build scripts
│   └── vite.config.js        # Vite build & proxy configuration
├── docker-compose.yml        # Multi-container orchestration
├── render.yaml               # 1-Click Render Deployment Blueprint
└── README.md
```

---

## Deploying to Render

This project includes a **`render.yaml`** blueprint that deploys the frontend and backend automatically.

### Option 1: Automatic 1-Click Blueprint (Recommended)
1. Push your code to GitHub.
2. Go to your **[Render Dashboard](https://dashboard.render.com)**.
3. Click **New +** > **Blueprint**.
4. Connect your GitHub repository (`AGENTIC-RAG_PROJECT`).
5. Render will automatically detect `render.yaml` and configure both services:
   - **`district-rag-backend`** (Web Service, Python)
   - **`district-rag-frontend`** (Static Site, React)
6. Add your environment variables in the Render Dashboard under backend settings:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
   - `SERPAPI_API_KEY`
   - `GNEWS_API_KEY`
   - `CLOUDFLARE_ACCOUNT_ID` & `CLOUDFLARE_API_TOKEN` (optional, for images)
   - `HF_TOKEN` (optional)
7. Click **Apply**. Render will build and deploy both services!

---

### Option 2: Manual Deployment on Render

#### 1. Backend Service
- **Type:** Web Service
- **Environment:** Python 3
- **Root Directory:** `backend`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Environment Variables:** Add `GROQ_API_KEY`, `GEMINI_API_KEY`, `SERPAPI_API_KEY`, etc.

#### 2. Frontend Service
- **Type:** Static Site
- **Root Directory:** `frontend`
- **Build Command:** `npm install && npm run build`
- **Publish Directory:** `dist`
- **Environment Variable:**
  - `VITE_API_BASE_URL`: `https://<your-backend-service-name>.onrender.com`

---

## Local Development

### 1. Backend
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).
