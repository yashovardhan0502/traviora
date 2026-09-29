import os
import sys
import uuid
import operator
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import certifi
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from typing import TypedDict, Annotated, Any
import json
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    AnyMessage 
)
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search, 
    aviation_mcp_call, 
    extract_destination, 
    forecast_mcp_search, 
    weather_mcp_search
)



# Environment
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL URL."
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")


DATABASE_URL = get_database_url()



# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)



# State
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + Guardrail State
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str
    
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New Budget + HITL State
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int



# Shared Helpers
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]



async def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)



def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }



# Supervisor Agent + Guardrail
async def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state["llm_calls"]

    guardrail_prompt = f"""
    Determine whether the following request belongs to travel planning or travel information. Valid requests can include destinations, flights, hotels, weather, budgets, visas, transportation, sightseeing, food, packing, or itineraries.

    Block clearly unrelated requests and requests asking for harmful or illegal instructions. Do not block a valid travel request merely because some details are missing.

    Return strict JSON only:
        {{
            "allowed": true,
            "reason": ""
        }}

    User Request:
    {query}
    """

    try:
        guardrail_raw = await _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1

    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed: 
        reason = guardrail_reason or (
            "Traviora can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, or itinerary."
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }
    

    supervisor_prompt = f"""
    You are the supervisor of a multi-agent travel-planning system.
    Choose only the specialist agents needed for the request.

    Available agents:
    - flight_agent: flights, airports, airlines, routes, airfare, or booking advice
    - hotel_agent: hotels, accommodation, neighborhoods, or places to stay
    - weather_agent: weather, climate, season, forecast, or packing advice
    - budget_agent: cost, affordability, price limits, or budget feasibility
    - itinerary_agent: creates the integrated travel plan and must always be included

    Return strict JSON only using this schema:
    {{
        "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
        "trip_constraints": {{
            "destination": "",
            "origin": "",
            "duration": "",
            "budget": "",
            "travel_style": "",
            "special_preferences": []
        }},
        "reasoning": ""
    }}

    User request:
    {query}
    """

    try:
        supervisor_raw = await _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # Itinerary agent should be there
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1

    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }



# Guardrail blocked response
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }



# Flight Agent
FLIGHT_AGENT_PROMPT = """
    You are an expert travel flight planner.

    User Query: {query}

    Based on the departure and destination cities, infer the most likely airports and common airlines.

    Provide concise travel guidance with:

    1. Departure airport (IATA code if known)
    2. Arrival airport (IATA code if known)
    3. Common airlines on this route
    4. Approximate flight duration
    5. Estimated airfare range
    6. Peak-season pricing warning (if relevant)
    7. Booking advice

    Keep the answer under 200 words.
    """

async def flight_agent(state: TravelState):
    print("\n INSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try: 
        
        prompt = FLIGHT_AGENT_PROMPT.format(
            query = query
        )

        response = await llm.ainvoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt)
            ]
        )

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Hotel Agent
async def hotel_agent(state: TravelState): 
    query = f"Best hotels for {state['user_query']}"
    

    try:
        hotel_results = await tavily_mcp_search(query)

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood guidance based on the destination and clearly label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }



# Weather Agent
async def weather_agent(state: TravelState):
    city = await extract_destination(state["user_query"])

    try:
        weather_data = await weather_mcp_search(city)
        forecast_data = await forecast_mcp_search(city)

        weather_results = f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }



# Budget Agent
async def budget_agent(state: TravelState):
    prompt = f"""
    Analyze whether this trip is realistic for the user's budget.

    User Query:
    {state['user_query']}

    Trip Constraints:
    {state.get('trip_constraints', {})}

    Flight Results:
    {state.get('flight_results', '')}

    Hotel Results:
    {state.get('hotel_results', '')}

    Weather Results:
    {state.get('weather_results', '')}

    Return:
    1. Estimated cost categories
    2. Budget risk areas
    3. Money-saving suggestions
    4. Overall feasibility

    If exact live prices are unavailable, clearly label estimates as approximate.
    """

    response=await llm.ainvoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt)
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget Assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Itinerary Agent
async def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a complete travel itinerary.

    User Query:
    {state['user_query']}

    Trip Constraints:
    {state.get('trip_constraints', {})}

    Flight Results:
    {state.get('flight_results', '')}

    Hotel Results:
    {state.get('hotel_results', '')}

    Weather Results:
    {state.get('weather_results', '')}

    Make the itinerary practical, budget-aware and easy to follow.
    """

    response = await llm.ainvoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request=(
        "Please review the generated draft itinerary. Approve it to create the final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



# Human-in-the-loop Approval
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. Langgraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }
 


# Final Agent
async def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decision while polishing it."
        )
    else:
        review_instruction=f"""
            The user requested a revision. Apply this feedback carefully:
            {state.get('human_feedback', '') or 'Improve the draft before finishing it.'}
        """

    final_prompt = f"""
    Generate the final travel response.

    Human Review:
    {review_instruction}

    User Request:
    {state["user_query"]}

    Supervisor Constraints:
    {state.get('trip_constraints', {})}

    Flights:
    {state.get('flight_results', '')}

    Hotels:
    {state.get('hotel_results', '')}

    Weather:
    {state.get('weather_results', '')}

    Budget Analysis:
    {state.get('budget_results', '')}

    Draft Itinerary:
    {state.get('itinerary', '')}

    Format the final answer:

    1. Trip Summary
    2. Flight Information
    3. Hotel Suggestions
    4. Weather Information
    5. Day-by-Day Itinerary
    6. Estimated Budget
    7. Final Recommendations

    Important:
    -> Be clear and practical.
    ->Mention that flight pricing may be unavailable if the live API doesn't provide prices.
    ->Keep the response useful for travel planning.
    """

    response = await llm.ainvoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



# Dynamic Supervisor Routing
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}



def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]



def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"



def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route



# Building the Graph
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)



# PostgreSQL Checkpointer
travel_graph = None
async_conn = None
checkpointer = None


async def init_graph():
    global travel_graph, async_conn, checkpointer

    if travel_graph is not None:
        return

    async_conn = await psycopg.AsyncConnection.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )

    checkpointer = AsyncPostgresSaver(async_conn)
    await checkpointer.setup()

    travel_graph = graph.compile(checkpointer=checkpointer)


async def close_graph():
    global async_conn

    if async_conn is not None:
        await async_conn.close()
        async_conn = None



# FastAPI-facing helpers
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result( result: dict[str, Any], thread_id: str,) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


async def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    global travel_graph
    if travel_graph is None:
        await init_graph()

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = await travel_graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


async def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    global travel_graph
    if travel_graph is None:
        await init_graph()

    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = await travel_graph.ainvoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)