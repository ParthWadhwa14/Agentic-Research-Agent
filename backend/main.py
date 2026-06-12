import asyncio
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_agent import generate_ppt_from_report, run_advanced_research
from media_search import collect_slide_images
from data_analysis_agent import analyze_data_files

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PDF_EXTENSIONS = {".pdf"}

app = FastAPI(
    title="Research Agentic Chatbot API",
    description="FastAPI backend for chat, research workflows, file uploads, reports, and PPT generation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://agentic-research-agent.vercel.app",
    ],
    allow_origin_regex=r"https://agentic-research-agent.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    service: str


class AgentResponse(BaseModel):
    id: str
    mode: str
    format: str
    answer: str
    artifacts: List[dict]


def save_uploads(files: Optional[List[UploadFile]]) -> List[Path]:
    saved_paths: List[Path] = []
    for file in files or []:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        destination = UPLOAD_DIR / safe_name
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(destination)
    return saved_paths


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_path(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


def is_data_path(path: Path) -> bool:
    return path.suffix.lower() in DATA_EXTENSIONS


def categorize_paths(paths: List[Path]) -> dict:
    categorized = {"pdf_paths": [], "image_paths": [], "data_paths": [], "source_files": []}
    for path in paths:
        suffix = path.suffix.lower()
        path_str = str(path)
        if is_pdf_path(path):
            categorized["pdf_paths"].append(path_str)
        elif is_image_path(path):
            categorized["image_paths"].append(path_str)
        elif is_data_path(path):
            categorized["data_paths"].append(path_str)
        else:
            categorized["source_files"].append(path_str)
    return categorized


TEXT_RESPONSE_TERMS = [
    "tell me",
    "explain",
    "summarize",
    "summary",
    "what is",
    "what are",
    "show me",
    "describe",
    "content",
    "contents",
    "above",
    "previous",
    "already created",
    "in chat",
    "answer in chat",
    "normal text answer",
    "not a downloadable",
    "do not download",
    "don't download",
]


def has_text_answer_intent(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in TEXT_RESPONSE_TERMS)


def has_explicit_output_request(query: str) -> bool:
    lowered = query.lower()
    if has_text_answer_intent(query):
        return False

    explicit_formats = ["pptx", "powerpoint", "slide deck", "latex", "tex file", ".tex", ".md", "text file", ".txt", "txt report"]
    output_actions = ["download", "export", "save", "create a file", "make a file", "generate a file"]
    report_actions = [
        "create report",
        "create a report",
        "generate report",
        "generate a report",
        "write report",
        "write a report",
        "make report",
        "make a report",
        "export report",
        "download report",
    ]
    return any(term in lowered for term in explicit_formats + output_actions + report_actions)


def fallback_output_decision(query: str, requested_format: str) -> Tuple[str, bool]:
    normalized_format = requested_format if requested_format != "auto" else infer_output_format(query)
    create_artifact = False if requested_format == "auto" and has_text_answer_intent(query) else should_create_artifact(requested_format, normalized_format, query)
    return normalized_format, create_artifact


def should_create_artifact(requested_format: str, normalized_format: str, query: str) -> bool:
    if requested_format != "auto":
        return True
    if normalized_format == "pptx":
        return wants_new_presentation_artifact(query) and not has_text_answer_intent(query)
    return has_explicit_output_request(query)


def infer_output_format(query: str, fallback: str = "plain text") -> str:
    lowered = query.lower()
    if has_text_answer_intent(query):
        return fallback
    if any(term in lowered for term in ["pptx", "powerpoint", "slide deck"]):
        return "pptx"
    if re.search(r"\b(make|create|generate|build|draft)\b.*\b(ppt|presentation|slides)\b|\b(ppt|presentation|slides)\b.*\b(make|create|generate|build|draft)\b", lowered):
        return "pptx"
    if any(term in lowered for term in ["latex", "tex file", ".tex"]):
        return "latex"
    if any(term in lowered for term in ["markdown", ".md"]):
        return "markdown"
    if any(term in lowered for term in ["plain text", "text file", ".txt", "txt report"]):
        return "plain text"
    return fallback


def llm_output_decision(query: str, requested_format: str, recent_messages: Optional[List[str]] = None) -> Tuple[str, bool]:
    """Lets the model decide whether the user wants a file or an in-chat answer."""
    if requested_format != "auto":
        return requested_format, True
    if not os.getenv("GROQ_API_KEY"):
        return fallback_output_decision(query, requested_format)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        recent_context = "\n".join(recent_messages or [])[-2500:]
        router = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        response = router.invoke([
            SystemMessage(content=(
                "Classify the newest user message for a chatbot backend. Return valid JSON only with keys "
                "`format` and `create_artifact`. `format` must be one of: plain text, markdown, latex, pptx. "
                "`create_artifact` is true only when the user wants a new downloadable/exported/saved file. "
                "If the user asks to tell, explain, summarize, show, inspect, discuss, or ask about content of an existing/above/previous PPT, report, or file, set create_artifact false and format plain text. "
                "Do not create artifacts merely because words like PPT, slides, Markdown, LaTeX, or file are mentioned. "
                "Create pptx only for requests to make/create/generate/build/draft/export/download a new PowerPoint/slide deck."
            )),
            HumanMessage(content=f"Recent context:\n{recent_context}\n\nNewest user message:\n{query}"),
        ])
        content = str(response.content).replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        output_format = str(decision.get("format", "plain text")).lower().strip()
        if output_format not in {"plain text", "markdown", "latex", "pptx"}:
            output_format = "plain text"
        create_artifact = bool(decision.get("create_artifact", False))
        return output_format, create_artifact
    except Exception:
        return fallback_output_decision(query, requested_format)


def write_text_artifact(run_id: str, content: str, format_type: str) -> None:
    extension_by_format = {
        "markdown": "md",
        "latex": "tex",
        "plain text": "txt",
    }
    extension = extension_by_format.get(format_type, "txt")
    artifact_path = OUTPUT_DIR / f"{run_id}.{extension}"
    formatted_content = format_text_artifact(content, format_type)
    artifact_path.write_text(formatted_content, encoding="utf-8")


def format_text_artifact(content: str, format_type: str) -> str:
    cleaned = sanitize_agent_response(content)
    if format_type in {"markdown", "latex"}:
        return cleaned + "\n"

    plain = cleaned
    plain = re.sub(r"(?m)^#{1,6}\s*", "", plain)
    plain = re.sub(r"\*\*(.*?)\*\*", r"\1", plain)
    plain = re.sub(r"__(.*?)__", r"\1", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"(?m)^\s*[-*]\s+", "• ", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()

    lines = [line.rstrip() for line in plain.splitlines()]
    title = "Generated Answer"
    for line in lines:
        stripped = line.strip(" #*-")
        if stripped:
            title = stripped[:90]
            break

    divider = "=" * min(max(len(title), 24), 90)
    body = "\n".join(lines).strip()
    return f"{title}\n{divider}\n\n{body}\n"


def artifact_ready_message(format_type: str) -> str:
    label_by_format = {
        "pptx": "PowerPoint presentation",
        "markdown": "Markdown report",
        "latex": "LaTeX report",
        "plain text": "text report",
    }
    label = label_by_format.get(format_type, "file")
    return f"Done — I created your {label}. You can download it below."


def build_artifacts(run_id: str, format_type: str) -> List[dict]:
    artifacts = []
    text_extensions = {
        "markdown": "md",
        "latex": "tex",
        "plain text": "txt",
    }
    if format_type in text_extensions:
        text_file = OUTPUT_DIR / f"{run_id}.{text_extensions[format_type]}"
        if text_file.exists():
            artifacts.append({
                "type": text_extensions[format_type],
                "name": text_file.name,
                "url": f"/api/download/{text_file.name}",
                "provenance": "generated locally by this app from the assistant response",
            })
    if format_type == "pptx":
        pptx_file = OUTPUT_DIR / f"{run_id}.pptx"
        if pptx_file.exists():
            artifacts.append({
                "type": "pptx",
                "name": pptx_file.name,
                "url": f"/api/download/{pptx_file.name}",
                "provenance": "generated locally by this app using python-pptx",
            })
    return artifacts


def build_file_artifacts(paths: List[str], provenance: str = "generated or collected by this app") -> List[dict]:
    artifacts = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            continue
        artifacts.append({
            "type": file_path.suffix.lower().lstrip(".") or "file",
            "name": file_path.name,
            "url": f"/api/download/{file_path.name}",
            "provenance": provenance,
        })
    return artifacts


def copy_artifacts_to_output(run_id: str, paths: List[str]) -> List[str]:
    final_paths: List[str] = []
    for path in paths:
        source = Path(path)
        if not source.exists() or not source.is_file():
            continue
        destination = OUTPUT_DIR / f"{run_id}_{source.name}"
        if source != destination:
            destination.write_bytes(source.read_bytes())
        final_paths.append(str(destination))
    return final_paths


def is_image_search_request(query: str) -> bool:
    lowered = query.lower()
    image_terms = ["image", "images", "picture", "pictures", "photo", "photos", "visuals"]
    find_terms = ["find", "search", "show", "give", "get", "suggest", "provide", "need", "want", "create", "generate", "make"]
    analysis_terms = ["uploaded image", "this image", "analyze image", "describe image"]
    if any(term in lowered for term in analysis_terms):
        return False
    return any(term in lowered for term in image_terms) and any(term in lowered for term in find_terms)


def image_search_response(run_id: str, query: str) -> AgentResponse:
    media_dir = OUTPUT_DIR / run_id
    paths = collect_slide_images(query, output_dir=media_dir, max_images=4)
    for path in paths:
        source = Path(path)
        destination = OUTPUT_DIR / f"{run_id}_{source.name}"
        if source != destination:
            destination.write_bytes(source.read_bytes())
            source.unlink(missing_ok=True)
    final_paths = [str(OUTPUT_DIR / f"{run_id}_{Path(path).name}") for path in paths]
    artifacts = build_file_artifacts(final_paths, "downloaded from web image search, not synthetically generated")
    if artifacts:
        answer = (
            "I can’t generate new images because this is a text-first agent, "
            "but I found a few relevant web images you can preview or download below."
        )
    else:
        answer = (
            "I can’t generate new images because this is a text-first agent. "
            "I tried to find relevant web images, but image search did not return usable results right now. "
            "I can still help write a strong image prompt or suggest visual ideas."
        )
    return AgentResponse(id=run_id, mode="chat", format="images", answer=answer, artifacts=artifacts)


def wants_new_presentation_artifact(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(
            r"\b(make|create|generate|build|draft|prepare|export|download)\b.*\b(ppt|pptx|powerpoint|presentation|slides|slide deck)\b|"
            r"\b(ppt|pptx|powerpoint|presentation|slides|slide deck)\b.*\b(make|create|generate|build|draft|prepare|export|download)\b",
            lowered,
        )
    )


def chunk_text(text: str, words_per_chunk: int = 18) -> List[str]:
    words = text.split(" ")
    if len(words) <= words_per_chunk:
        return [text]
    chunks = []
    for index in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[index:index + words_per_chunk]) + (" " if index + words_per_chunk < len(words) else ""))
    return chunks


def is_artifact_origin_question(query: str) -> bool:
    lowered = query.lower()
    artifact_terms = ["graph", "graphs", "chart", "charts", "image", "images", "ppt", "presentation", "file", "artifact", "above"]
    origin_terms = ["made by you", "generated by you", "created by you", "downloaded", "from net", "from the net", "internet", "where did", "source", "origin"]
    return any(term in lowered for term in artifact_terms) and any(term in lowered for term in origin_terms)


def is_app_meta_question(query: str) -> bool:
    lowered = query.lower()
    meta_terms = ["you", "your", "this app", "this chatbot", "above", "previous", "generated", "downloaded", "artifact", "file"]
    return any(term in lowered for term in meta_terms) and not any(
        term in lowered for term in ["research report", "deep research", "cite sources", "latest", "current news"]
    )


def answer_artifact_origin_question(artifact_context: str) -> str:
    if not artifact_context.strip():
        return (
            "I do not have artifact metadata for the files shown above in this request. "
            "If you mean the charts produced after CSV/Excel analysis, those are generated locally by this app from the uploaded dataset, not downloaded from the internet."
        )
    return (
        "The graphs/charts shown above were generated locally by this app from your uploaded data, not downloaded from the internet, "
        "when their artifact metadata says they came from local data analysis. Web image artifacts are labeled separately as downloaded from web image search.\n\n"
        f"{artifact_context}"
    )


async def build_agent_response(
    query: str,
    mode: str,
    format_type: str,
    conversation_summary: str,
    long_term_context: str,
    artifact_context: str,
    recent_messages: str,
    files: Optional[List[UploadFile]],
) -> AgentResponse:
    run_id = uuid.uuid4().hex
    normalized_mode = mode.lower().strip()
    requested_format = format_type.lower().strip()
    saved_paths = save_uploads(files)
    path_groups = categorize_paths(saved_paths)

    try:
        parsed_recent_messages = json.loads(recent_messages)
        if not isinstance(parsed_recent_messages, list):
            parsed_recent_messages = []
    except json.JSONDecodeError:
        parsed_recent_messages = []

    normalized_format, create_artifact = llm_output_decision(query, requested_format, parsed_recent_messages)

    try:
        if is_artifact_origin_question(query):
            return AgentResponse(
                id=run_id,
                mode="chat",
                format="plain text",
                answer=answer_artifact_origin_question(artifact_context),
                artifacts=[],
            )

        if is_image_search_request(query):
            return await asyncio.to_thread(image_search_response, run_id, query)

        if normalized_mode == "chat":
            data_analysis = {"summary": "", "chart_paths": []}
            if path_groups["data_paths"]:
                data_analysis = await asyncio.to_thread(
                    analyze_data_files,
                    path_groups["data_paths"],
                    OUTPUT_DIR / run_id,
                    query,
                )
            answer = await asyncio.to_thread(
                run_chat_agent,
                query,
                saved_paths,
                parsed_recent_messages,
                long_term_context,
                str(data_analysis.get("summary", "")),
            )
            answer = sanitize_agent_response(answer)
            if create_artifact and normalized_format == "pptx":
                slide_images = list(data_analysis.get("chart_paths", []))
                slide_images.extend(await asyncio.to_thread(
                    collect_slide_images,
                    query,
                    OUTPUT_DIR,
                    None,
                    4,
                ))
                await asyncio.to_thread(
                    generate_ppt_from_report,
                    report=answer,
                    query=query,
                    output_path=str(OUTPUT_DIR / f"{run_id}.pptx"),
                    chart_paths=slide_images,
                )
                answer = artifact_ready_message(normalized_format)
            elif create_artifact:
                write_text_artifact(run_id, answer, normalized_format)
                answer = artifact_ready_message(normalized_format)
            extra_artifacts = build_file_artifacts(
                copy_artifacts_to_output(run_id, list(data_analysis.get("chart_paths", []))),
                "generated locally by this app from the uploaded dataset using pandas/matplotlib",
            )
        else:
            data_analysis = {"summary": "", "chart_paths": []}
            if path_groups["data_paths"]:
                data_analysis = await asyncio.to_thread(
                    analyze_data_files,
                    path_groups["data_paths"],
                    OUTPUT_DIR / run_id,
                    query,
                )
            pptx_path = str(OUTPUT_DIR / f"{run_id}.pptx")
            answer = await asyncio.to_thread(
                run_advanced_research,
                query=query,
                format_type=normalized_format,
                pdf_paths=path_groups["pdf_paths"],
                image_paths=path_groups["image_paths"],
                data_paths=path_groups["data_paths"],
                source_files=path_groups["source_files"],
                pptx_path=pptx_path,
                conversation_summary="\n\n".join(section for section in [conversation_summary, long_term_context] if section.strip()),
                recent_messages=parsed_recent_messages,
            )
            answer = sanitize_agent_response(answer)
            if create_artifact and normalized_format != "pptx":
                write_text_artifact(run_id, answer, normalized_format)
                answer = artifact_ready_message(normalized_format)
            elif create_artifact:
                answer = artifact_ready_message(normalized_format)
            extra_artifacts = build_file_artifacts(
                copy_artifacts_to_output(run_id, list(data_analysis.get("chart_paths", []))),
                "generated locally by this app from the uploaded dataset using pandas/matplotlib",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentResponse(
        id=run_id,
        mode=normalized_mode,
        format=normalized_format,
        answer=answer,
        artifacts=(build_artifacts(run_id, normalized_format) if create_artifact else []) + extra_artifacts,
    )


def run_chat_agent(
    query: str,
    saved_paths: List[Path],
    recent_messages: Optional[List[str]] = None,
    long_term_context: str = "",
    data_context: str = "",
) -> str:
    try:
        from basic_agent import run_research_agent
    except Exception as exc:
        raise RuntimeError(f"Normal chat agent could not be loaded: {exc}") from exc

    image_path = next((str(path) for path in saved_paths if is_image_path(path)), None)
    doc_path = next((str(path) for path in saved_paths if is_pdf_path(path)), None)
    return sanitize_agent_response(
        run_research_agent(
            query_text=query,
            image_path=image_path,
            doc_path=doc_path,
            recent_messages=recent_messages,
            long_term_context=long_term_context,
            data_context=data_context,
        )
    )


def sanitize_agent_response(answer: str) -> str:
    replacements = {
        "using the 'graph_tool' function": "with code",
        "using the graph_tool function": "with code",
        "'graph_tool' function": "code/charting capability",
        "graph_tool": "code/charting capability",
        "'web_search'": "search",
        "web_search": "search",
    }
    cleaned = answer
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"(?im)^\s*<\s*search\b[^>]*>\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*</\s*search\s*>\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*<\s*tool\b[^>]*>\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*</\s*tool\s*>\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="research-agent-api")


@app.post("/api/agent", response_model=AgentResponse)
async def run_agent(
    query: str = Form(...),
    mode: str = Form("chat"),
    format_type: str = Form("auto"),
    conversation_summary: str = Form(""),
    long_term_context: str = Form(""),
    artifact_context: str = Form(""),
    recent_messages: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(default=None),
) -> AgentResponse:
    return await build_agent_response(
        query=query,
        mode=mode,
        format_type=format_type,
        conversation_summary=conversation_summary,
        long_term_context=long_term_context,
        artifact_context=artifact_context,
        recent_messages=recent_messages,
        files=files,
    )


@app.post("/api/agent/stream")
async def stream_agent(
    query: str = Form(...),
    mode: str = Form("chat"),
    format_type: str = Form("auto"),
    conversation_summary: str = Form(""),
    long_term_context: str = Form(""),
    artifact_context: str = Form(""),
    recent_messages: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(default=None),
) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Received request'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'Running agent workflow'})}\n\n"
        try:
            response = await build_agent_response(
                query=query,
                mode=mode,
                format_type=format_type,
                conversation_summary=conversation_summary,
                long_term_context=long_term_context,
                artifact_context=artifact_context,
                recent_messages=recent_messages,
                files=files,
            )
            for chunk in chunk_text(response.answer):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.015)
            yield f"data: {json.dumps({'type': 'final', 'answer': response.answer, 'artifacts': response.artifacts})}\n\n"
        except Exception as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(detail)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/download/{filename}")
def download_artifact(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path, filename=safe_name)
