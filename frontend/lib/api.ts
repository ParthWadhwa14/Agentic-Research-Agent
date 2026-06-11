export type AgentMode = "chat" | "research";
export type OutputFormat = "auto" | "markdown" | "plain text" | "latex" | "pptx";

export type Artifact = {
  type: string;
  name: string;
  url: string;
};

export type AgentResponse = {
  id: string;
  mode: AgentMode;
  format: OutputFormat;
  answer: string;
  artifacts: Artifact[];
};

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function runAgentRequest(input: {
  query: string;
  mode: AgentMode;
  format?: OutputFormat;
  files: File[];
  recentMessages?: string[];
  conversationSummary?: string;
}): Promise<AgentResponse> {
  const formData = new FormData();
  formData.append("query", input.query);
  formData.append("mode", input.mode);
  formData.append("format_type", input.format ?? "auto");
  formData.append("conversation_summary", input.conversationSummary ?? "");
  formData.append("recent_messages", JSON.stringify(input.recentMessages ?? []));
  input.files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/api/agent`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown API error" }));
    throw new Error(error.detail ?? "Agent request failed");
  }

  return response.json();
}

export async function runAgentStream(
  input: {
    query: string;
    mode: AgentMode;
    format?: OutputFormat;
    files: File[];
    recentMessages?: string[];
    conversationSummary?: string;
  },
  handlers: {
    onStatus?: (message: string) => void;
    onChunk?: (content: string) => void;
    onFinal?: (response: Pick<AgentResponse, "answer" | "artifacts">) => void;
    onError?: (message: string) => void;
  }
): Promise<void> {
  const formData = new FormData();
  formData.append("query", input.query);
  formData.append("mode", input.mode);
  formData.append("format_type", input.format ?? "auto");
  formData.append("conversation_summary", input.conversationSummary ?? "");
  formData.append("recent_messages", JSON.stringify(input.recentMessages ?? []));
  input.files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/api/agent/stream`, {
    method: "POST",
    body: formData
  });

  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => ({ detail: "Unknown API error" }));
    throw new Error(error.detail ?? "Agent stream failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(6));
      if (payload.type === "status") handlers.onStatus?.(payload.message);
      if (payload.type === "chunk") handlers.onChunk?.(payload.content);
      if (payload.type === "final") handlers.onFinal?.({ answer: payload.answer, artifacts: payload.artifacts ?? [] });
      if (payload.type === "error") handlers.onError?.(payload.message);
    }
  }
}

export function artifactUrl(path: string) {
  return `${API_BASE}${path}`;
}
