import os
import uuid
import operator

import certifi
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    AnyMessage,
)
from langchain_groq import ChatGroq

from mcp_client import tavily_mcp_search, aviation_mcp_call, extract_destination, forecast_mcp_search, weather_mcp_search



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
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str



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
        "messages": [
            AIMessage(content="Flight recommendations generated.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



#Hotel Agent
async def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = await tavily_mcp_search(query)

    summary = await llm.ainvoke([
        SystemMessage(content="Summarize hotel search into 5 concise bullet points with hotel names, approximate price range, and location."),
        HumanMessage(content=str(hotel_results)[:3000])
    ])

    hotel_results = summary.content

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



#Weather Agent
async def weather_agent(state: TravelState):
    city = await extract_destination(state["user_query"])

    weather_data=await weather_mcp_search(city)
    forecast_data=await forecast_mcp_search(city)

    return {
        "weather_results": f"""
        Current Weather: {weather_data}

        Forecast: {forecast_data}
        """,
        "messages": [
            AIMessage(content="Weather information fetched")
        ]
    }




#Itinerary Agent
async def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state["user_query"]}

Flight Results:
{state["flight_results"]}

Hotel Results:
{state["hotel_results"]}

Weather Results:
{state["weather_results"]}

Make the itinerary practical, budget-aware and easy to follow.
"""

    response = await llm.ainvoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response.

User Request:
{state["user_query"]}

Flight Summary:
{state["flight_results"][:800]}

Hotel Summary:
{state["hotel_results"][:800]}

Weather:
{state["weather_results"]}

Itinerary Summary:
{state["itinerary"][:800]}

Format:

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
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -----------------------------
# Graph
# -----------------------------

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)




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


# -----------------------------
# Main Runner
# -----------------------------


async def run_travel_agent(user_input: str, thread_id: str | None = None):
    global travel_graph

    if travel_graph is None:
        await init_graph()

    if thread_id is None:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = await travel_graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_input)],
           
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return {
        "thread_id": thread_id,
        "answer": result["messages"][-1].content,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }