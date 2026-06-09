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

export function artifactUrl(path: string) {
  return `${API_BASE}${path}`;
}
