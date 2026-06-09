import asyncio
import json
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_agent import generate_ppt_from_report, run_advanced_research

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
    ],
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


def has_explicit_output_request(query: str) -> bool:
    lowered = query.lower()
    if any(term in lowered for term in ["explain", "summarize", "summary", "what is", "tell me about", "understand"]):
        return False

    explicit_formats = [
        "ppt",
        "pptx",
        "powerpoint",
        "presentation",
        "slide deck",
        "slides",
        "latex",
        "tex file",
        ".tex",
        "markdown",
        ".md",
        "plain text",
        "text file",
        ".txt",
        "txt report",
    ]
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


def infer_output_format(query: str, fallback: str = "plain text") -> str:
    lowered = query.lower()
    if any(term in lowered for term in ["ppt", "pptx", "powerpoint", "presentation", "slide deck", "slides"]):
        return "pptx"
    if any(term in lowered for term in ["latex", "tex file", ".tex"]):
        return "latex"
    if any(term in lowered for term in ["markdown", ".md"]):
        return "markdown"
    if any(term in lowered for term in ["plain text", "text file", ".txt", "txt report"]):
        return "plain text"
    return fallback


def write_text_artifact(run_id: str, content: str, format_type: str) -> None:
    extension_by_format = {
        "markdown": "md",
        "latex": "tex",
        "plain text": "txt",
    }
    extension = extension_by_format.get(format_type, "txt")
    artifact_path = OUTPUT_DIR / f"{run_id}.{extension}"
    artifact_path.write_text(content, encoding="utf-8")


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
            })
    if format_type == "pptx":
        pptx_file = OUTPUT_DIR / f"{run_id}.pptx"
        if pptx_file.exists():
            artifacts.append({
                "type": "pptx",
                "name": pptx_file.name,
                "url": f"/api/download/{pptx_file.name}",
            })
    return artifacts


def run_chat_agent(query: str, saved_paths: List[Path], recent_messages: Optional[List[str]] = None) -> str:
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
    recent_messages: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(default=None),
) -> AgentResponse:
    run_id = uuid.uuid4().hex
    normalized_mode = mode.lower().strip()
    requested_format = format_type.lower().strip()
    normalized_format = requested_format if requested_format != "auto" else infer_output_format(query)
    saved_paths = save_uploads(files)

    try:
        parsed_recent_messages = json.loads(recent_messages)
        if not isinstance(parsed_recent_messages, list):
            parsed_recent_messages = []
    except json.JSONDecodeError:
        parsed_recent_messages = []

    try:
        if normalized_mode == "chat":
            answer = await asyncio.to_thread(run_chat_agent, query, saved_paths, parsed_recent_messages)
            answer = sanitize_agent_response(answer)
            if normalized_format == "pptx":
                await asyncio.to_thread(
                    generate_ppt_from_report,
                    report=answer,
                    query=query,
                    output_path=str(OUTPUT_DIR / f"{run_id}.pptx"),
                )
                answer = artifact_ready_message(normalized_format)
            elif has_explicit_output_request(query):
                write_text_artifact(run_id, answer, normalized_format)
                answer = artifact_ready_message(normalized_format)
        else:
            paths = categorize_paths(saved_paths)
            pptx_path = str(OUTPUT_DIR / f"{run_id}.pptx")
            answer = await asyncio.to_thread(
                run_advanced_research,
                query=query,
                format_type=normalized_format,
                pdf_paths=paths["pdf_paths"],
                image_paths=paths["image_paths"],
                data_paths=paths["data_paths"],
                source_files=paths["source_files"],
                pptx_path=pptx_path,
                conversation_summary=conversation_summary,
                recent_messages=parsed_recent_messages,
            )
            answer = sanitize_agent_response(answer)
            if normalized_format != "pptx":
                write_text_artifact(run_id, answer, normalized_format)
                answer = artifact_ready_message(normalized_format)
            else:
                answer = artifact_ready_message(normalized_format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentResponse(
        id=run_id,
        mode=normalized_mode,
        format=normalized_format,
        answer=answer,
        artifacts=build_artifacts(run_id, normalized_format),
    )


@app.post("/api/agent/stream")
async def stream_agent(
    query: str = Form(...),
    mode: str = Form("chat"),
    format_type: str = Form("auto"),
    files: Optional[List[UploadFile]] = File(default=None),
) -> StreamingResponse:
    saved_paths = save_uploads(files)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Received request'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': 'Running agent workflow'})}\n\n"
        try:
            run_id = uuid.uuid4().hex
            requested_format = format_type.lower().strip()
            normalized_format = requested_format if requested_format != "auto" else infer_output_format(query)
            if mode.lower().strip() == "chat":
                answer = await asyncio.to_thread(run_chat_agent, query, saved_paths)
                answer = sanitize_agent_response(answer)
                if normalized_format == "pptx":
                    await asyncio.to_thread(
                        generate_ppt_from_report,
                        report=answer,
                        query=query,
                        output_path=str(OUTPUT_DIR / f"{run_id}.pptx"),
                    )
                    answer = artifact_ready_message(normalized_format)
                elif has_explicit_output_request(query):
                    write_text_artifact(run_id, answer, normalized_format)
                    answer = artifact_ready_message(normalized_format)
                artifacts = build_artifacts(run_id, normalized_format)
            else:
                paths = categorize_paths(saved_paths)
                answer = await asyncio.to_thread(
                    run_advanced_research,
                    query=query,
                    format_type=normalized_format,
                    pdf_paths=paths["pdf_paths"],
                    image_paths=paths["image_paths"],
                    data_paths=paths["data_paths"],
                    source_files=paths["source_files"],
                    pptx_path=str(OUTPUT_DIR / f"{run_id}.pptx"),
                )
                answer = sanitize_agent_response(answer)
                if normalized_format != "pptx":
                    write_text_artifact(run_id, answer, normalized_format)
                    answer = artifact_ready_message(normalized_format)
                else:
                    answer = artifact_ready_message(normalized_format)
                artifacts = build_artifacts(run_id, normalized_format)
            yield f"data: {json.dumps({'type': 'final', 'answer': answer, 'artifacts': artifacts})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/download/{filename}")
def download_artifact(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path, filename=safe_name)
