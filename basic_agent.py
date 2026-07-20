

import os 
import base64
import io
import mimetypes
import operator 
import re
import math
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Sequence, Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv()

# LangChain & LangGraph core imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model providers & Vector stores
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper

# LangGraph orchestration
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class BasicAgent(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    state: str 
    context: Dict[str, str]

# ==========================================
# 2. MODEL INITIALIZATIONS (Corrected Identifiers)
# ==========================================
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Using valid, high-performance production endpoints
llm_vision = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.2
)

llm_code = ChatGroq(
    model="openai/gpt-oss-120b", 
    api_key=GROQ_API_KEY
)

# ==========================================
# 3. HELPER FUNCTIONS & STANDALONE METHODS
# ==========================================
def encode_image(image_path: str) -> str:
    """Reads a local image file and converts it to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prepare_image_data_url(image_path: str, max_side: int = 1600, jpeg_quality: int = 85) -> str:
    """Builds a vision-model-compatible data URL, resizing if Pillow is available."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return f"data:{mime_type};base64,{encode_image(image_path)}"

def get_comprehensive_image_info(image_path: str) -> str:
    """Analyzes a local image using standard LangChain multimodal messaging structures."""
    if not os.path.exists(image_path):
        return f"Error: No image found at path '{image_path}'."

    master_prompt = (
        "Analyze the uploaded image carefully for a chatbot answer. Return a concise but useful structured analysis:\n"
        "1. Overall description: what the image shows and what is happening.\n"
        "2. Important objects/people/data: list visible key elements.\n"
        "3. Text/OCR: transcribe all visible text exactly; if none, say None.\n"
        "4. If it is a chart/table/screenshot/document, explain the data, UI, or content.\n"
        "5. Limitations: mention anything unclear or unreadable."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": master_prompt},
            {
                "type": "image_url",
                "image_url": {"url": prepare_image_data_url(image_path)},
            },
        ]
    )
    
    try:
        response = llm_vision.invoke([message])
        content = str(response.content).strip()
        return content or "Image was received, but the vision model returned an empty analysis."
    except Exception as e:
        return f"Image was uploaded at {image_path}, but vision analysis failed: {str(e)}"

DOCUMENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with", "you", "your",
    "me", "my", "about", "explain", "summarize", "summary", "report", "document", "pdf", "file",
}


def tokenize_for_retrieval(text: str) -> List[str]:
    """Tokenizes text for lightweight local PDF retrieval without downloading embeddings."""
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
        if token not in DOCUMENT_STOPWORDS
    ]


def chunk_pdf_documents(docs: List[Document]) -> List[Document]:
    """Creates overlapping page-aware chunks for stronger PDF grounding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=220,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = index + 1
    return chunks


def document_overview(docs: List[Document], max_chars: int = 3500) -> str:
    """Keeps the beginning and section-like lines so explanation requests get document structure."""
    page_count = len(docs)
    full_text = "\n".join(doc.page_content for doc in docs)
    section_lines = []
    for line in full_text.splitlines():
        clean_line = re.sub(r"\s+", " ", line).strip()
        if not clean_line:
            continue
        looks_like_heading = (
            len(clean_line) <= 90
            and (
                clean_line.isupper()
                or re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.)\s+[A-Z]", clean_line)
                or clean_line.lower().startswith(("abstract", "introduction", "conclusion", "results", "method"))
            )
        )
        if looks_like_heading:
            section_lines.append(clean_line)
        if len(section_lines) >= 18:
            break

    opening = clean_text(full_text, max_chars=max_chars)
    sections = "\n".join(f"- {line}" for line in section_lines)
    return (
        f"Document overview:\n"
        f"- Pages loaded: {page_count}\n"
        f"- Detected sections/headings:\n{sections or '- No clear headings detected.'}\n\n"
        f"Opening/context excerpt:\n{opening}"
    )


def retrieve_context(query: str, docs: List[Document], max_chars: int = 18000) -> str:
    """Retrieves relevant PDF chunks with lexical scoring and broad coverage for explanation tasks."""
    chunks = chunk_pdf_documents(docs)
    if not chunks:
        return "No readable text chunks were extracted from the PDF."

    query_tokens = tokenize_for_retrieval(query)
    explanation_request = bool(
        re.search(r"\b(explain|summarize|summary|overview|understand|what is this|tell me about)\b", query, re.IGNORECASE)
    )
    if explanation_request or not query_tokens:
        query_tokens = tokenize_for_retrieval(query + " abstract introduction conclusion results findings recommendation")

    doc_freq = Counter()
    chunk_tokens = []
    for chunk in chunks:
        tokens = tokenize_for_retrieval(chunk.page_content)
        chunk_tokens.append(tokens)
        doc_freq.update(set(tokens))

    scored_chunks = []
    total_chunks = len(chunks)
    for chunk, tokens in zip(chunks, chunk_tokens):
        token_counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if token not in token_counts:
                continue
            idf = math.log((total_chunks + 1) / (doc_freq[token] + 1)) + 1
            score += token_counts[token] * idf

        page_number = int(chunk.metadata.get("page", 0)) + 1 if str(chunk.metadata.get("page", "")).isdigit() else 0
        if explanation_request and page_number in {1, 2}:
            score += 1.5
        if explanation_request and re.search(r"\b(abstract|introduction|conclusion|findings|results|recommendation)\b", chunk.page_content, re.IGNORECASE):
            score += 2.0
        scored_chunks.append((score, chunk))

    selected = [chunk for score, chunk in sorted(scored_chunks, key=lambda item: item[0], reverse=True)[:10] if score > 0]
    if explanation_request:
        selected_indices = {chunk.metadata.get("chunk_index") for chunk in selected}
        for chunk in chunks[:3]:
            if chunk.metadata.get("chunk_index") not in selected_indices:
                selected.append(chunk)
        for chunk in chunks[-2:]:
            if chunk.metadata.get("chunk_index") not in selected_indices:
                selected.append(chunk)

    if not selected:
        selected = chunks[:8]

    selected = sorted(selected[:14], key=lambda chunk: chunk.metadata.get("chunk_index", 0))
    context_parts = [document_overview(docs)]
    used_chars = len(context_parts[0])
    for chunk in selected:
        page = chunk.metadata.get("page")
        page_label = f"Page {int(page) + 1}" if isinstance(page, int) else "Page unknown"
        excerpt = clean_text(chunk.page_content, 1500)
        block = f"\n\n[{page_label}, chunk {chunk.metadata.get('chunk_index')}]\n{excerpt}"
        if used_chars + len(block) > max_chars:
            break
        context_parts.append(block)
        used_chars += len(block)

    return "\n".join(context_parts)


def clean_text(value: Any, max_chars: int = 600) -> str:
    """Normalizes snippets so search context stays compact and readable."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_chars].rstrip()


def needs_live_information(query: str) -> bool:
    """Detects requests that should not be answered from model memory alone."""
    lowered = query.lower()
    live_terms = [
        "current",
        "currently",
        "right now",
        "today",
        "latest",
        "recent",
        "now",
        "weather",
        "forecast",
        "temperature",
        "time in",
        "date in",
        "this year",
        "this month",
        "this week",
        "tomorrow",
        "yesterday",
        "next year",
        "last year",
        "news",
        "price",
        "stock",
        "score",
        "match",
        "matches",
        "fixture",
        "fixtures",
        "schedule",
        "cricket",
        "series",
        "tournament",
        "election",
        "release",
        "available",
        "who is the",
        "where is the",
    ]
    return any(term in lowered for term in live_terms)


def build_current_datetime_context() -> str:
    """Provides a reliable clock anchor so relative dates resolve correctly."""
    local_time = datetime.now().astimezone()
    utc_time = datetime.now(ZoneInfo("UTC"))
    india_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    return "\n".join([
        f"Current server time: {local_time.strftime('%Y-%m-%d %I:%M %p %Z')}",
        f"Current UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Current India time: {india_time.strftime('%Y-%m-%d %I:%M %p %Z')} (Asia/Kolkata)",
        f"Current year: {india_time.year}",
    ])


def should_use_web_search(query: str) -> bool:
    """Chooses web grounding for factual questions likely to be stale or source-sensitive."""
    lowered = query.lower()
    if needs_live_information(query):
        return True
    search_triggers = [
        "search",
        "google",
        "look up",
        "find out",
        "according to",
        "source",
        "cite",
        "citation",
        "compare",
        "best",
        "top",
        "review",
        "paper",
        "research",
        "dataset",
        "statistics",
        "market",
        "company",
        "startup",
        "government",
        "law",
        "policy",
    ]
    return any(term in lowered for term in search_triggers)


def is_time_or_date_query(query: str) -> bool:
    """Detects clock/calendar requests that can be answered without web search."""
    lowered = query.lower()
    return bool(re.search(r"\b(time|date|day)\b", lowered)) and not bool(
        re.search(r"\b(weather|forecast|temperature|news|price|stock|score)\b", lowered)
    )


def build_time_context(query: str) -> str:
    """Returns exact time context for common location/time-zone questions."""
    lowered = query.lower()
    contexts = [build_current_datetime_context()]

    requested_zones = []
    if "india" in lowered or "ist" in lowered:
        requested_zones.append(("India", "Asia/Kolkata"))
    if "berlin" in lowered or "germany" in lowered:
        requested_zones.append(("Berlin, Germany", "Europe/Berlin"))

    for label, zone_name in requested_zones:
        local_time = datetime.now(ZoneInfo(zone_name))
        contexts.append(f"Current time in {label}: {local_time.strftime('%Y-%m-%d %I:%M %p %Z')} ({zone_name})")

    return "\n".join(contexts)


def build_search_queries(query: str) -> List[str]:
    """Creates a small set of targeted Serper queries for better retrieval."""
    clean_query = clean_text(query, 220)
    current_year = datetime.now(ZoneInfo("Asia/Kolkata")).year
    year_resolved_query = re.sub(r"\bthis year\b", str(current_year), clean_query, flags=re.IGNORECASE)
    year_resolved_query = re.sub(r"\bnext year\b", str(current_year + 1), year_resolved_query, flags=re.IGNORECASE)
    year_resolved_query = re.sub(r"\blast year\b", str(current_year - 1), year_resolved_query, flags=re.IGNORECASE)
    queries = [year_resolved_query]
    if re.search(r"\b(india|indian).*\b(cricket|match|matches|fixture|fixtures|schedule|series)\b|\b(cricket|match|matches|fixture|fixtures|schedule|series).*\b(india|indian)\b", query, re.IGNORECASE):
        queries = [
            f"India men's cricket schedule remaining international matches {current_year} BCCI",
            f"site:bcci.tv international fixtures India cricket {current_year}",
            f"site:icc-cricket.com India fixtures {current_year} cricket",
        ]
    elif re.search(r"\b(weather|forecast|temperature)\b", query, re.IGNORECASE):
        queries.append(f"{year_resolved_query} official current weather")
    elif re.search(r"\b(news|latest|recent|today|current)\b", query, re.IGNORECASE):
        queries.append(f"{year_resolved_query} latest news today")
    elif re.search(r"\b(date|festival|holiday|diwali|deepavali|eid|christmas|thanksgiving)\b", query, re.IGNORECASE):
        queries.append(f"{year_resolved_query} date India official")
    elif re.search(r"\b(compare|comparison|best|top|review)\b", query, re.IGNORECASE):
        queries.append(f"{year_resolved_query} comparison reviews")
    elif re.search(r"\b(research|paper|study|statistics|market)\b", query, re.IGNORECASE):
        queries.append(f"{year_resolved_query} sources statistics")

    deduped = []
    for item in queries:
        if item and item.lower() not in {existing.lower() for existing in deduped}:
            deduped.append(item)
    return deduped[:3]


def format_serper_results(payload: Dict[str, Any], query: str, max_items: int = 6) -> str:
    """Turns raw Serper JSON into compact cited evidence for the LLM."""
    lines = [f"Search query: {query}"]

    answer_box = clean_text(payload.get("answerBox") or payload.get("knowledgeGraph"), 900)
    if answer_box:
        lines.append(f"Direct result: {answer_box}")

    organic_results = payload.get("organic") or payload.get("news") or payload.get("places") or []
    for index, item in enumerate(organic_results[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        title = clean_text(item.get("title"), 180)
        snippet = clean_text(item.get("snippet") or item.get("description"), 500)
        link = clean_text(item.get("link"), 260)
        date = clean_text(item.get("date"), 80)
        source = clean_text(item.get("source"), 120)
        details = []
        if date:
            details.append(f"date: {date}")
        if source:
            details.append(f"source: {source}")
        meta = f" ({'; '.join(details)})" if details else ""
        if title or snippet or link:
            lines.append(f"{index}. {title}{meta}\n   {snippet}\n   URL: {link}")

    return "\n".join(lines)


def run_serper_search(query: str) -> str:
    """Runs Serper directly and returns source-rich snippets, not vague summaries."""
    if not os.getenv("SERPER_API_KEY"):
        return "Web search unavailable: SERPER_API_KEY is not set in the backend environment."

    sections = []
    for search_query in build_search_queries(query):
        try:
            search = GoogleSerperAPIWrapper(k=8, gl="us", hl="en")
            payload = search.results(search_query)
            sections.append(format_serper_results(payload, search_query))
        except Exception as exc:
            sections.append(f"Search failed for '{search_query}': {exc}")

    return "\n\n".join(section for section in sections if section.strip())


def build_live_search_context(query: str) -> str:
    """Runs direct current/search grounding without exposing tool calls to the model."""
    if not should_use_web_search(query):
        return build_current_datetime_context()

    sections = [build_time_context(query)]
    if is_time_or_date_query(query):
        return "\n".join(section for section in sections if section.strip())

    sections.append(run_serper_search(query))

    return "\n\n".join(section for section in sections if section.strip())

# ==========================================
# 4. TOOLS DEFINITION
# ==========================================
@tool
def web_search(query: str) -> str:
    """Performs a source-rich Serper web search and returns cited snippets."""
    return run_serper_search(query)

@tool
def graph_tool(query: str) -> str:
    """This tool generates clean Python analysis/graphing code using matplotlib, pandas, and numpy."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant specialized in creating pure Python code blocks using matplotlib, pandas, and numpy. Return ONLY executable raw code. Do NOT enclose in markdown blocks, do not add text explanations, and do not include conversational fluff."),
        ("human", "{query}")
    ])
    chain = prompt | llm_code
    response = chain.invoke({"query": query})  
    return response.content

# Keep tools available for future expansion, but do not bind them to normal chat.
# Tool binding caused provider-level function-call failures for ordinary code/PPT prompts.
tools_list = [web_search, graph_tool]
llm_with_tools = llm_code

# ==========================================
# 5. LANGGRAPH NODE FUNCTIONS
# ==========================================
def decision_agent(state: BasicAgent) -> Dict[str, Any]:
    """Evaluates request history along with structural vector and image analysis context."""
    
    # Extract the text from the newest message string
    user_query = state["messages"][-1].content
    pdf_info = state["context"].get("pdf_data", "No PDF Data provided.")
    img_info = state["context"].get("image_data", "No Image Data provided.")
    live_info = state["context"].get("live_data", "")
    memory_info = state["context"].get("memory_data", "")
    data_info = state["context"].get("data_data", "")

    system_instructions = (
        "You are a direct, natural chat assistant. Answer like a capable human collaborator: concise, warm, and useful. "
        "Never say you are a large language model, never explain that you do not have feelings, and never use generic AI-boilerplate disclaimers. "
        "Use recent conversation history and LONG-TERM MEMORY as memory. If the user shared their name, preferences, projects, or decisions earlier, remember and use them naturally. "
        "Treat long-term memory as user-provided context, but do not expose raw memory blocks unless the user asks what you remember. "
        "When DATA ANALYSIS CONTEXT is provided, answer as a practical data analyst using the uploaded dataset first. "
        "Discuss trends, outliers, correlations, limitations, and chart insights from that dataset. Do not substitute web search for uploaded data analysis. "
        "Be self-aware about this app's artifacts: CSV/Excel analysis charts are generated locally from uploaded datasets using code, PPT/report files are generated locally by the app, and web image artifacts are downloaded from web image search. "
        "If the user asks whether above graphs/charts were generated by you or downloaded from the net, answer directly from artifact/conversation context; do not produce a research report. "
        "For current, latest, weather, date, time, factual comparison, source-sensitive, or research questions, ground the answer in the LIVE/SEARCH CONTEXT below. "
        "When search results are available, synthesize them into a clear answer and include source names or URLs for important factual claims. "
        "Do not invent facts, dates, prices, weather, or statistics that are not in the context. Do not reuse old dates from model memory. "
        "For sports fixtures, schedules, prices, and availability, only use the LIVE/SEARCH CONTEXT; if exact dates are not present, say that clearly instead of guessing. "
        "For India cricket schedules, prefer official BCCI/ICC source results and list only matches that are still upcoming relative to the current date. "
        "If live search is unavailable, clearly say what exact data you can verify and what you cannot verify, then answer only from stable knowledge if safe. "
        "Never output XML/HTML-style tool markup such as <search>, <search query=...>, <tool>, or any hidden action tags. "
        "Do not say that you are searching; simply answer from the provided LIVE/SEARCH CONTEXT. "
        "For greetings or small talk, reply briefly and naturally, then invite the user forward. "
        "Never expose implementation details, internal tool names, function names, framework names, or phrases like 'using the graph_tool function'. "
        "If the user refers to something 'above' but the needed details are not available in the current conversation or provided files, say briefly what is missing and ask for the specific values/data. "
        "When asking for missing details, do not add generic offers, long explanations, or tool references. "
        "You are a text-first research/chat agent running inside an app that can chat, research, read uploaded PDFs/images, find relevant web images, and export reports/PPTs. "
        "You do not generate or create new synthetic images. If the user asks for image generation, say clearly that this app is a text agent; it can find relevant web images, help write an image prompt, or analyze an uploaded image instead. "
        "You can answer questions, write code, create presentation content when asked, and produce calculations or charts when the user provides enough data. "
        "When document context is provided and the user asks to explain, summarize, or understand the report, give a complete in-chat explanation. "
        "Structure report explanations with: big picture, section-by-section explanation, key findings, important details, limitations, and what the user should take away. "
        "Cite page numbers from the document context when possible. Do not create a downloadable file unless the user explicitly asks to export/download/save one. "
        "For PPT/presentation requests in chat, create a polished slide-by-slide outline with strong titles, concise bullets, suggested visuals, and speaker notes. "
        "Below is supplemental parsed context from document indices and computer vision processes that you should utilize.\n\n"
        f"--- LONG-TERM MEMORY ---\n{memory_info or 'No saved user memory available.'}\n\n"
        f"--- DATA ANALYSIS CONTEXT ---\n{data_info or 'No uploaded dataset analysis available.'}\n\n"
        f"--- LIVE/SEARCH CONTEXT ---\n{live_info or 'No live/current context required or available.'}\n\n"
        f"--- DOCUMENT INDEX CONTEXT ---\n{pdf_info}\n\n"
        f"--- VISION EXTRACTION CONTEXT ---\n{img_info}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instructions),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm_with_tools
    try:
        response = chain.invoke({"messages": state["messages"]})
    except Exception as exc:
        failed_generation = getattr(exc, "body", {}).get("error", {}).get("failed_generation") if hasattr(exc, "body") else None
        if failed_generation:
            response = AIMessage(content=failed_generation)
        else:
            raise
    
    # Return the updated dictionary to match state schema appending
    return {"messages": [response]}

# ==========================================
# 6. GRAPH TOPOLOGY COMPILATION
# ==========================================
graph_builder = StateGraph(BasicAgent)

# Add processing nodes
graph_builder.add_node("decision_agent", decision_agent)

# Route workflow transitions
graph_builder.add_edge(START, "decision_agent")
graph_builder.add_edge("decision_agent", END)

# Compile into runnable application entity
compiled_agent_app = graph_builder.compile()

# ==========================================
# 7. EXPORTABLE MASTER INVOCATION FUNCTION
# ==========================================
def _messages_from_recent_history(recent_messages: Optional[Sequence[str]], query_text: str) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for item in list(recent_messages or [])[-8:]:
        role, _, content = item.partition(":")
        clean_content = content.strip() if content else item.strip()
        if not clean_content:
            continue
        if role.strip().lower() == "assistant":
            messages.append(AIMessage(content=clean_content))
        else:
            messages.append(HumanMessage(content=clean_content))
    messages.append(HumanMessage(content=query_text))
    return messages


def run_research_agent(
    query_text: str,
    image_path: Optional[str] = None,
    doc_path: Optional[str] = None,
    recent_messages: Optional[Sequence[str]] = None,
    long_term_context: str = "",
    data_context: str = "",
) -> str:
    """
    Main entry point function. Handles pipeline ingestion, document indexing,
    vision extraction, state construction, and graph processing loops.
    """
    pdf_context_data = "None"
    image_context_data = "None"

    # Handle document indexing if a path exists
    if doc_path and os.path.exists(doc_path):
        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(doc_path)
            docs = loader.load()
            pdf_context_data = retrieve_context(query_text, docs)
        except Exception as e:
            pdf_context_data = f"Failed to ingest document index: {str(e)}"

    # Handle vision processing if a path exists
    if image_path and os.path.exists(image_path):
        image_size = os.path.getsize(image_path)
        image_mime = mimetypes.guess_type(image_path)[0] or "unknown"
        image_context_data = (
            f"Uploaded image detected:\n"
            f"- Path: {image_path}\n"
            f"- MIME/type: {image_mime}\n"
            f"- Size: {image_size} bytes\n\n"
            f"Vision analysis:\n{get_comprehensive_image_info(image_path)}"
        )

    # Structure initial inputs into standard BasicAgent state schema dictionary
    initial_state = {
        "messages": _messages_from_recent_history(recent_messages, query_text),
        "state": "init_processing",
        "context": {
            "pdf_data": pdf_context_data,
            "image_data": image_context_data,
            "live_data": build_current_datetime_context() if data_context else build_live_search_context(query_text),
            "memory_data": long_term_context,
            "data_data": data_context,
        }
    }

    # Execute graph processing loop across nodes
    final_output_state = compiled_agent_app.invoke(initial_state)
    
    # Extract final conversational text content string from the final message
    return final_output_state["messages"][-1].content












































