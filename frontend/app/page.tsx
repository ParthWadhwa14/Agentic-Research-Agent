"use client";

import { FormEvent, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AgentMode, Artifact, artifactUrl, runAgentRequest } from "../lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  artifacts?: Artifact[];
};

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [mode, setMode] = useState<AgentMode>("chat");
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const recentMessages = useMemo(
    () => messages.slice(-8).map((message) => `${message.role}: ${message.content.slice(0, 1000)}`),
    [messages]
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim() || isLoading) return;

    const userQuery = query.trim();
    const attachedFiles = files;
    setMessages((current) => [...current, { role: "user", content: userQuery }]);
    setQuery("");
    setError("");
    setIsLoading(true);

    try {
      const response = await runAgentRequest({
        query: userQuery,
        mode,
        files: attachedFiles,
        recentMessages
      });
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, artifacts: response.artifacts }
      ]);
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Something went wrong";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen flex-col bg-[#f7f7f8] text-[#1f2328]">
      <header className="sticky top-0 z-10 border-b border-black/10 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-black text-sm font-semibold text-white">AI</div>
            <div>
              <h1 className="text-sm font-semibold">Research Chatbot</h1>
              <p className="text-xs text-gray-500">Chat, research, files, reports</p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as AgentMode)}
              className="rounded-lg border border-gray-300 bg-white px-2 py-1 outline-none"
            >
              <option value="research">Research</option>
              <option value="chat">Chat</option>
            </select>
          </div>
        </div>
      </header>

      <section className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-4">
        {messages.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center text-center">
            <h2 className="text-3xl font-semibold tracking-tight">How can I help you today?</h2>
          </div>
        ) : (
          <div className="flex-1 space-y-6 py-8">
            {messages.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))}
            {isLoading && <AssistantTyping />}
            {error && <div className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</div>}
          </div>
        )}
      </section>

      <footer className="sticky bottom-0 border-t border-black/10 bg-[#f7f7f8]/90 px-4 py-4 backdrop-blur">
        <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
          {files.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {files.map((file) => (
                <span key={`${file.name}-${file.size}`} className="rounded-full bg-white px-3 py-1 text-xs text-gray-600 shadow-sm">
                  {file.name}
                </span>
              ))}
            </div>
          )}

          <div className="flex items-end gap-2 rounded-3xl border border-gray-300 bg-white p-2 shadow-sm">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-xl text-gray-500 hover:bg-gray-100"
              aria-label="Attach files"
            >
              +
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.png,.jpg,.jpeg,.webp,.gif,.bmp,.csv,.xlsx,.xls"
              className="hidden"
              onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
            />
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Message Research Chatbot..."
              rows={1}
              className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm outline-none placeholder:text-gray-400"
            />
            <button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-black text-white disabled:bg-gray-300"
              aria-label="Send message"
            >
              ↑
            </button>
          </div>
          <p className="mt-2 text-center text-xs text-gray-400">Research Chatbot can make mistakes. Check important outputs.</p>
        </form>
      </footer>
    </main>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[82%] rounded-3xl px-4 py-3 text-sm leading-7 ${isUser ? "bg-black text-white" : "bg-transparent text-gray-900"}`}>
        <MarkdownMessage content={message.content} />
        {message.artifacts && message.artifacts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.artifacts.map((artifact) => (
              <a
                key={artifact.url}
                href={artifactUrl(artifact.url)}
                target="_blank"
                rel="noreferrer"
                className="rounded-full border border-gray-300 bg-white px-3 py-1 text-xs font-medium text-gray-900 hover:bg-gray-100"
              >
                Download {artifact.type.toUpperCase()}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="prose-chat">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="rounded bg-gray-200 px-1.5 py-0.5 text-[0.85em] text-gray-900" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="my-4 overflow-x-auto rounded-2xl border border-gray-200 bg-[#0d1117] p-4 text-sm text-gray-100 shadow-sm">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function AssistantTyping() {
  return (
    <div className="flex justify-start">
      <div className="rounded-3xl bg-transparent px-4 py-3 text-sm text-gray-500">Thinking...</div>
    </div>
  );
}
