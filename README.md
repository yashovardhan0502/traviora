# ✈️ Traviora — Multi-Agent Travel Planner

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-1.2.2-blue?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/LangChain-1.3.2-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/FastAPI-0.136.3-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Groq-Fast_Inference-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
  <img src="https://img.shields.io/badge/MCP-Model_Context_Protocol-8A2BE2?style=for-the-badge" alt="MCP" />
  <img src="https://img.shields.io/badge/PostgreSQL-Checkpointer-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge" alt="License" />
</p>

---

## 🌟 Overview

**Traviora** is an enterprise-grade, multi-agent AI travel orchestration engine designed to autonomously research, evaluate, and craft personalized end-to-end travel itineraries.

Powered by **LangGraph**, **Groq LLMs**, **Model Context Protocol (MCP)**, and **PostgreSQL state persistence**, Traviora routes user travel goals through specialized AI agents for flight intelligence, live hotel search, weather forecasting, and budget feasibility. It incorporates a **Safety Guardrail Engine** and a native **Human-in-the-Loop (HITL)** review gate, allowing travelers to inspect and refine draft itineraries before final plan generation.

---

## ✨ Key Features

- 🧠 **Supervisor Agent & Dynamic Routing**: Evaluates user prompts, extracts trip constraints (destination, origin, duration, budget, travel style, preferences), and dynamically plans agent execution order.
- 🛡️ **Built-in Safety & Domain Guardrails**: Automatically detects and blocks off-topic, malicious, or non-travel queries before invoking downstream tools.
- 🌐 **Model Context Protocol (MCP) Multi-Server Integration**:
  - **Tavily MCP Server** (`Streamable HTTP`): Real-time web intelligence and hotel search.
  - **AviationStack MCP Server** (`stdio` via `uvx`): Live aviation data and flight route analysis.
  - **Custom Weather FastMCP Server** (`stdio`): Live weather conditions and multi-day meteorological forecasts via OpenWeatherMap API.
- 👤 **Human-in-the-Loop (HITL) Workflow**: Employs LangGraph `interrupt()` to present draft itineraries for traveler approval or structured feedback revisions.
- 💾 **Stateful Checkpointing with PostgreSQL**: Full session persistence via `AsyncPostgresSaver`, enabling thread-level state restoration and conversational continuity.
- ⚡ **Ultra-Low Latency Inference**: Uses **Groq Cloud** (`openai/gpt-oss-120b`) for real-time agent coordination and content synthesis.
- 🎨 **Modern Interactive Web UI**: Sleek glassmorphic web interface built with FastAPI & Jinja2, live agent execution visualizer, Markdown renderer, and one-click PDF export via `html2pdf.js`.

---

## 📐 System Architecture

### 1. Multi-Agent StateGraph Workflow

Traviora coordinates a dynamic execution pipeline where the Supervisor inspects the user intent and delegates tasks to domain specialists:

```mermaid
flowchart TD
    Start([🚀 User Query]) --> SupervisorNode[🧠 Supervisor Agent & Guardrail]
    
    SupervisorNode -->|Query Blocked / Off-topic| GuardrailBlock[🛡️ Guardrail Blocked Agent]
    GuardrailBlock --> EndNode([🏁 Return Guardrail Notice])

    SupervisorNode -->|Query Approved| RouteCondition{Dynamic Agent Routing}

    RouteCondition -->|Flight Included| FlightAgent[✈️ Flight Agent\n- Route Analysis & Airports\n- IATA Codes & Airfare Ranges]
    RouteCondition -->|Hotel Included| HotelAgent[🏨 Hotel Agent\n- Tavily MCP Live Search\n- Accommodations & Neighborhoods]
    RouteCondition -->|Weather Included| WeatherAgent[🌦️ Weather Agent\n- Custom FastMCP Server\n- Live Conditions & Forecasts]
    RouteCondition -->|Budget Included| BudgetAgent[💰 Budget Agent\n- Cost Breakdown\n- Risk Assessment & Feasibility]
    
    FlightAgent --> HotelAgent
    HotelAgent --> WeatherAgent
    WeatherAgent --> BudgetAgent
    BudgetAgent --> ItineraryAgent[🗓️ Itinerary Agent\n- Synthesizes all Specialist Data\n- Generates Draft Itinerary]

    ItineraryAgent --> HITLNode[👤 Human-in-the-Loop Gate\nlanggraph.types.interrupt]

    HITLNode -->|User Feedback / Revision| FinalAgent[✨ Final Response Agent\n- Polishes Complete Itinerary\n- Applies Revision Feedback]
    HITLNode -->|User Approved| FinalAgent

    FinalAgent --> EndGraph([🎉 Final Comprehensive Travel Plan])

    classDef agent fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef gate fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef blocked fill:#4c0519,stroke:#f43f5e,stroke-width:2px,color:#fff;
    classDef terminal fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff;

    class SupervisorNode,FlightAgent,HotelAgent,WeatherAgent,BudgetAgent,ItineraryAgent,FinalAgent agent;
    class HITLNode gate;
    class GuardrailBlock blocked;
    class Start,EndNode,EndGraph terminal;
```

---

### 2. Model Context Protocol (MCP) Multi-Server Architecture

Traviora utilizes the `MultiServerMCPClient` from `langchain-mcp-adapters` to bridge the LangGraph agents with distinct tool providers:

```mermaid
flowchart LR
    subgraph TravioraCore ["⚙️ Traviora Backend"]
        Backend[backend.py / mcp_client.py]
        MultiMCP[MultiServerMCPClient]
    end

    subgraph MCPServers ["🌐 MCP Server Ecosystem"]
        TavilyMCP["🔍 Tavily MCP Server\n(Streamable HTTP Transport)\n• tavily_search"]
        AviationMCP["🛫 AviationStack MCP Server\n(Stdio Transport via uvx)\n• aviationstack-mcp"]
        WeatherMCP["☀️ Custom Weather MCP Server\n(Stdio FastMCP Transport)\n• get_current_weather\n• get_forecast"]
    end

    subgraph ExternalAPIs ["☁️ External Provider APIs"]
        TavilyAPI[Tavily Search API]
        AviationAPI[AviationStack API]
        OpenWeatherAPI[OpenWeatherMap API]
    end

    Backend --> MultiMCP
    MultiMCP -->|HTTP / SSE| TavilyMCP --> TavilyAPI
    MultiMCP -->|Stdio Stream| AviationMCP --> AviationAPI
    MultiMCP -->|Stdio Stream| WeatherMCP --> OpenWeatherAPI

    classDef core fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef mcp fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef ext fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#fff;

    class Backend,MultiMCP core;
    class TavilyMCP,AviationMCP,WeatherMCP mcp;
    class TavilyAPI,AviationAPI,OpenWeatherAPI ext;
```

---

### 3. Human-in-the-Loop (HITL) Execution Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 Traveler / Client
    participant Web as 💻 FastAPI Web UI
    participant LG as ⚡ LangGraph Engine
    participant DB as 🗄️ PostgreSQL (AsyncPostgresSaver)
    participant LLM as 🤖 Groq LLM & MCP Tools

    User->>Web: Submits travel prompt (e.g., "7 days Japan trip")
    Web->>LG: POST /api/travel (user_query, thread_id)
    LG->>DB: Checkpoint initial state
    LG->>LLM: Supervisor guardrail & agent selection
    LG->>LLM: Execute selected agents (Flight, Hotel, Weather, Budget)
    LG->>LLM: Itinerary Agent creates draft itinerary
    LG->>LG: interrupt() triggers -> Pause execution
    LG->>DB: Save state checkpoint (thread_id)
    LG-->>Web: Return draft itinerary + requires_approval=True
    Web-->>User: Display draft itinerary + Approval/Revision Card

    alt Traveler Approves Plan
        User->>Web: Clicks "Approve & Generate Final"
        Web->>LG: POST /api/travel/approve (thread_id, approved=True)
    else Traveler Requests Changes
        User->>Web: Enters revision feedback and clicks "Revise"
        Web->>LG: POST /api/travel/approve (thread_id, approved=False, feedback="...")
    end

    LG->>DB: Load checkpoint state for thread_id
    LG->>LG: Command(resume={approved, feedback})
    LG->>LLM: Final Agent synthesizes polished travel plan
    LG->>DB: Checkpoint completed state
    LG-->>Web: Return final comprehensive itinerary
    Web-->>User: Render formatted itinerary (PDF Export & Copy available)
```

---

## 🤖 Agent Roles & Specialist Breakdown

| Agent Name | Role & Responsibility | Tools & Data Sources |
| :--- | :--- | :--- |
| **Supervisor & Guardrail** | Validates travel intent, filters malicious prompts, extracts trip constraints, and dynamically routes active agents. | LLM (`openai/gpt-oss-120b`), Structured JSON Extraction |
| **Flight Specialist** | Analyzes departure/arrival airports (IATA codes), major airlines, durations, peak-season alerts, and estimated airfare ranges. | LLM Prompts & AviationStack MCP Integration |
| **Hotel Specialist** | Identifies top accommodations, neighborhood safety, amenities, and proximity to major sights. | **Tavily MCP Search** (Streamable HTTP) |
| **Weather Specialist** | Extracts destination city and retrieves real-time weather metrics, humidity, wind speed, and 5-step forecasts. | **Custom Weather FastMCP Server** + OpenWeatherMap API |
| **Budget Specialist** | Calculates cost breakdowns (transit, lodging, food, activities), highlights financial risk areas, and suggests money-saving tips. | LLM Analytical Synthesis |
| **Itinerary Specialist** | Synthesizes all specialist findings into a cohesive, structured day-by-day draft itinerary. | LLM Multi-Agent Aggregation |
| **Human Approval Gate** | Halts execution state and exposes draft plan for traveler review via LangGraph `interrupt()`. | LangGraph Native Checkpoint Interrupt |
| **Final Polish Specialist**| Incorporates traveler feedback or approval to finalize a clean, comprehensive, production-ready travel itinerary. | LLM Final Polishing & Structuring |

---

## 📂 Project Structure

```
traviora/
├── app.py                         # FastAPI web application, route handlers & lifecycle
├── backend.py                     # LangGraph StateGraph, supervisor routing, agents & checkpointer
├── mcp_client.py                  # MultiServerMCPClient configuration & tool wrappers
├── custom_weather_mcp_server.py   # FastMCP standalone server for OpenWeatherMap API
├── requirements.txt               # Project Python dependencies
├── Dockerfile                     # Containerization specification
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git ignore specifications
├── LICENSE                        # Apache 2.0 License
├── templates/
│   └── index.html                 # Jinja2 frontend template with glassmorphic layout
├── static/
│   ├── style.css                  # Modern responsive CSS styles, animations & theme tokens
│   └── script.js                  # Frontend client logic, API calls, marked.js & html2pdf integration
└── test.py                        # MCP tool verification script
```

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration & State Machine**: [LangGraph](https://github.com/langchain-ai/langgraph) (`1.2.2`), [LangChain](https://github.com/langchain-ai/langchain) (`1.3.2`)
- **LLM Engine**: [Groq Cloud](https://groq.com/) via `langchain-groq` (`openai/gpt-oss-120b`)
- **Model Context Protocol**: `langchain-mcp-adapters` (`0.3.0`), `mcp` (`1.28.1`), `FastMCP`
- **Database & Persistence**: [PostgreSQL](https://www.postgresql.org/) with `psycopg3`, `psycopg-pool`, and `langgraph-checkpoint-postgres` (`3.1.0`)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) (`0.136.3`) + [Uvicorn](https://www.uvicorn.org/) (`0.48.0`)
- **External APIs & Tools**:
  - [Tavily AI Search](https://tavily.com/)
  - [OpenWeatherMap API](https://openweathermap.org/)
  - [AviationStack API](https://aviationstack.com/)
- **Frontend Technologies**: Vanilla JavaScript (ES6+), Modern CSS3 Glassmorphism, [Marked.js](https://marked.js.org/), [html2pdf.js](https://ekoopmans.github.io/html2pdf.js/)

---

## ⚙️ Environment Variables Configuration

Create a `.env` file in the project root directory and provide the required API credentials:

```env
# Groq Cloud API Key (Required for LLM inference)
GROQ_API_KEY=gsk_your_groq_api_key_here

# PostgreSQL Database Connection String (Render / Supabase / Neon / Local)
DATABASE_URL=postgresql://user:password@host:5432/dbname?sslmode=require

# Tavily AI Search API Key (For live web & hotel research)
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# OpenWeatherMap API Key (For weather forecasts)
OPENWEATHER_API_KEY=your_openweathermap_api_key_here

# AviationStack API Key (Optional / For live flight data)
AVIATIONSTACK_API_KEY=your_aviationstack_api_key_here
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python 3.11** installed
- **PostgreSQL** instance (Local or hosted on Render/Supabase/Neon)
- **Node / uvx** (Optional, for AviationStack stdio MCP transport)

### 1. Clone the Repository
```bash
git clone https://github.com/yashovardhan0502/traviora.git
cd traviora
```

### 2. Create and Activate Virtual Environment

**Using Conda:**
```bash
conda create -n travel python=3.11 -y
conda activate travel
```

**Using Python venv:**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file and update it with your credentials:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env

# Linux / macOS:
cp .env.example .env
```


### 5. Launch the Application
Start the FastAPI application with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🐳 Docker Deployment

You can build and run Traviora using Docker:

### 1. Build the Docker Image
```bash
docker build -t traviora:latest .
```

### 2. Run the Container
```bash
docker run -d \
  --name traviora-app \
  -p 8000:8000 \
  --env-file .env \
  traviora:latest
```

Navigate to `http://localhost:8000` to access the application.

---

## 📡 API Reference

### 1. Plan Trip (Initial Draft & HITL Trigger)
- **Endpoint**: `POST /api/travel`
- **Payload**:
```json
{
  "message": "Plan a 7-day budget trip to Tokyo, Japan from Mumbai under 1.5 lakhs.",
  "thread_id": "optional_custom_thread_id"
}
```
- **Response**:
```json
{
  "success": true,
  "thread_id": "user_4a2c9f81...",
  "answer": "### Draft Itinerary for Tokyo...",
  "requires_approval": true,
  "approval_request": "Please review the generated draft itinerary...",
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {
    "destination": "Tokyo, Japan",
    "origin": "Mumbai, India",
    "duration": "7 days",
    "budget": "1.5 lakhs",
    "travel_style": "budget",
    "special_preferences": []
  },
  "guardrail_allowed": true
}
```

### 2. Resume & Approve / Revise Plan
- **Endpoint**: `POST /api/travel/approve`
- **Payload (Approval)**:
```json
{
  "thread_id": "user_4a2c9f81...",
  "approved": true,
  "feedback": ""
}
```
- **Payload (Revision Request)**:
```json
{
  "thread_id": "user_4a2c9f81...",
  "approved": false,
  "feedback": "Please add one free day for shopping in Akihabara and reduce hotel expense."
}
```
- **Response**:
```json
{
  "success": true,
  "thread_id": "user_4a2c9f81...",
  "answer": "### Final Travel Plan for Tokyo...",
  "requires_approval": false,
  "approved": true
}
```

### 3. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "ok",
  "message": "TripMate AI API is running",
  "features": [
    "supervisor_agent",
    "input_guardrail",
    "human_in_the_loop"
  ]
}
```

---

## 🧪 Testing

To test the MCP connection and Tavily search integration directly:
```bash
python test.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Crafted with ❤️ for intelligent, seamless travel planning.
</p>
