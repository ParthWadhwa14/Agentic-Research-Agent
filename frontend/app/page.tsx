"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { GoogleAuthProvider, onAuthStateChanged, signInWithEmailAndPassword, signInWithPopup, signOut, type User } from "firebase/auth";
import { AgentMode, Artifact, artifactUrl, runAgentStream } from "../lib/api";
import { auth, authReady } from "../lib/firebase";
import { friendlyFirebaseError } from "../lib/firebaseErrors";
import {
  addMessage,
  buildLongTermContext,
  createConversation,
  deleteConversation,
  extractProjectCandidate,
  extractMemoryCandidates,
  getConversation,
  getUserProfile,
  listConversations,
  listMemories,
  listMessages,
  listProjects,
  makeConversationSummary,
  makeTitle,
  saveExtractedMemory,
  saveProjectState,
  updateConversationMeta,
  upsertUserProfile,
  type StoredConversation,
  type StoredMemory,
  type StoredMessage,
  type StoredProject,
  type StoredUser
} from "../lib/chatStorage";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  artifacts?: Artifact[];
};

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeStorageConversationIdRef = useRef("");
  const [authUser, setAuthUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<StoredUser | null>(null);
  const [conversations, setConversations] = useState<StoredConversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [activeSummary, setActiveSummary] = useState("");
  const [memories, setMemories] = useState<StoredMemory[]>([]);
  const [projects, setProjects] = useState<StoredProject[]>([]);
  const [mode, setMode] = useState<AgentMode>("chat");
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBooting, setIsBooting] = useState(true);
  const [isChatsLoading, setIsChatsLoading] = useState(false);
  const [loadingConversationId, setLoadingConversationId] = useState("");
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [error, setError] = useState("");
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
  const [streamStatus, setStreamStatus] = useState("");

  useEffect(() => {
    return onAuthStateChanged(auth, async (user) => {
      setAuthUser(user);
      setIsBooting(false);
      if (!user) {
        setProfile(null);
        setConversations([]);
        setMessages([]);
        setMemories([]);
        setProjects([]);
        setActiveConversationId("");
        activeStorageConversationIdRef.current = "";
        return;
      }
      const fallbackProfile = profileFromFirebaseUser(user);
      setProfile(readCache<StoredUser>(cacheKey(user.uid, "profile")) ?? fallbackProfile);
      setConversations(readCache<StoredConversation[]>(cacheKey(user.uid, "conversations")) ?? []);
      setMemories(readCache<StoredMemory[]>(cacheKey(user.uid, "memories")) ?? []);
      setProjects(readCache<StoredProject[]>(cacheKey(user.uid, "projects")) ?? []);
      void safeStorage(() => upsertUserProfile(user), undefined);
      void refreshUserData(user.uid, fallbackProfile);
    });
  }, []);

  async function refreshUserData(userId: string, fallbackProfile?: StoredUser) {
    setIsChatsLoading(true);
    const [nextProfile, nextConversations] = await Promise.all([
      safeStorage(() => getUserProfile(userId), fallbackProfile ?? null),
      safeStorage(() => listConversations(userId), readCache<StoredConversation[]>(cacheKey(userId, "conversations")) ?? [])
    ]);
    setProfile(nextProfile ?? fallbackProfile ?? null);
    setConversations(nextConversations);
    writeCache(cacheKey(userId, "profile"), nextProfile ?? fallbackProfile ?? null);
    writeCache(cacheKey(userId, "conversations"), nextConversations);
    setIsChatsLoading(false);

    const [nextMemories, nextProjects] = await Promise.all([
      safeStorage(() => listMemories(userId), readCache<StoredMemory[]>(cacheKey(userId, "memories")) ?? []),
      safeStorage(() => listProjects(userId), readCache<StoredProject[]>(cacheKey(userId, "projects")) ?? [])
    ]);
    setMemories(nextMemories);
    setProjects(nextProjects);
    writeCache(cacheKey(userId, "memories"), nextMemories);
    writeCache(cacheKey(userId, "projects"), nextProjects);
  }

  async function loadConversation(conversationId: string) {
    if (!authUser) return;
    setError("");
    setActiveConversationId(conversationId);
    activeStorageConversationIdRef.current = conversationId.startsWith("local-") ? "" : conversationId;
    const cachedMessages = readCache<ChatMessage[]>(cacheKey(authUser.uid, `messages:${conversationId}`));
    if (cachedMessages) setMessages(cachedMessages);
    if (conversationId.startsWith("local-")) {
      setActiveSummary("");
      return;
    }
    setLoadingConversationId(conversationId);
    const [conversation, storedMessages] = await Promise.all([
      safeStorage(() => getConversation(conversationId), null),
      safeStorage(() => listMessages(conversationId), [])
    ]);
    setLoadingConversationId("");
    setActiveSummary(conversation?.summary ?? "");
    const nextMessages = storedMessages.map((message: StoredMessage) => ({
        role: message.role,
        content: message.content,
        artifacts: message.artifacts ?? []
      }));
    setMessages(nextMessages);
    writeCache(cacheKey(authUser.uid, `messages:${conversationId}`), nextMessages);
  }

  async function handleSelectConversation(conversationId: string) {
    setIsMobileSidebarOpen(false);
    await loadConversation(conversationId);
  }

  async function handleNewChat() {
    if (!authUser) return;
    setIsMobileSidebarOpen(false);
    const conversationId = `local-${Date.now()}`;
    activeStorageConversationIdRef.current = "";
    setActiveConversationId(conversationId);
    setActiveSummary("");
    setMessages([]);
    writeCache(cacheKey(authUser.uid, "conversations"), [
      {
        id: conversationId,
        user_id: authUser.uid,
        title: "New chat",
        summary: ""
      },
      ...conversations
    ]);
    setConversations((current) => [
      {
        id: conversationId,
        user_id: authUser.uid,
        title: "New chat",
        summary: ""
      },
      ...current
    ]);
  }

  async function handleDeleteConversation(conversationId: string) {
    if (!authUser || isLoading) return;
    let nextConversations: StoredConversation[] = [];
    setConversations((current) => {
      nextConversations = current.filter((conversation) => conversation.id !== conversationId);
      writeCache(cacheKey(authUser.uid, "conversations"), nextConversations);
      return nextConversations;
    });
    removeCache(cacheKey(authUser.uid, `messages:${conversationId}`));
    if (!conversationId.startsWith("local-")) {
      void safeStorage(() => deleteConversation(conversationId), undefined);
    }
    if (conversationId === activeConversationId) {
      const nextId = nextConversations[0]?.id ?? "";
      if (nextId) await loadConversation(nextId);
      else {
        setActiveConversationId("");
        activeStorageConversationIdRef.current = "";
        setActiveSummary("");
        setMessages([]);
      }
    }
  }

  const recentMessages = useMemo(
    () =>
      messages.slice(-8).map((message) => {
        const artifacts = message.artifacts?.length
          ? `\nArtifacts: ${message.artifacts
              .map((artifact) => `${artifact.name} (${artifact.type}${artifact.provenance ? `, ${artifact.provenance}` : ""})`)
              .join("; ")}`
          : "";
        return `${message.role}: ${message.content.slice(0, 1000)}${artifacts}`;
      }),
    [messages]
  );

  const artifactContext = useMemo(() => {
    const artifactLines = messages
      .flatMap((message) => message.artifacts ?? [])
      .slice(-12)
      .map((artifact) => `- ${artifact.name} (${artifact.type}): ${artifact.provenance ?? "artifact returned by this app"}`);
    return artifactLines.length ? `Recent artifacts shown in this chat:\n${artifactLines.join("\n")}` : "";
  }, [messages]);

  const longTermContext = useMemo(
    () => buildLongTermContext({ user: profile, memories, projects, conversationSummary: activeSummary }),
    [profile, memories, projects, activeSummary]
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!authUser || !query.trim() || isLoading) return;

    const userQuery = query.trim();
    const attachedFiles = files;
    const isFirstMessage = messages.length === 0;
    const assistantIndex = messages.length + 1;

    setMessages((current) => [...current, { role: "user", content: userQuery }, { role: "assistant", content: "" }]);
    setQuery("");
    setError("");
    setStreamStatus("");
    setIsLoading(true);

    try {
      const conversationId = activeConversationId || `local-${Date.now()}`;
      const storageConversationId = activeStorageConversationIdRef.current;
      setActiveConversationId(conversationId);

      await runAgentStream(
        {
          query: userQuery,
          mode,
          files: attachedFiles,
          recentMessages,
          conversationSummary: activeSummary,
          longTermContext,
          artifactContext
        },
        {
          onStatus: setStreamStatus,
          onChunk: (content) => {
            setMessages((current) =>
              current.map((message, index) =>
                index === assistantIndex ? { ...message, content: `${message.content}${content}` } : message
              )
            );
          },
          onFinal: (response) => {
            const nextSummary = makeConversationSummary(activeSummary, userQuery, response.answer);
            const completedMessages: ChatMessage[] = [
              ...messages,
              { role: "user", content: userQuery },
              { role: "assistant", content: response.answer, artifacts: response.artifacts }
            ];
            setMessages((current) =>
              current.map((message, index) =>
                index === assistantIndex
                  ? { ...message, content: response.answer, artifacts: response.artifacts }
                  : message
              )
            );
            setActiveSummary(nextSummary);
            setConversations((current) => {
              const title = isFirstMessage ? makeTitle(userQuery) : current.find((item) => item.id === conversationId)?.title ?? makeTitle(userQuery);
              const nextConversation = { id: conversationId, user_id: authUser.uid, title, summary: nextSummary };
              const withoutCurrent = current.filter((item) => item.id !== conversationId);
              const nextConversations = [nextConversation, ...withoutCurrent];
              writeCache(cacheKey(authUser.uid, "conversations"), nextConversations);
              writeCache(cacheKey(authUser.uid, `messages:${conversationId}`), completedMessages);
              return nextConversations;
            });
            void persistChatTurn({
              userId: authUser.uid,
              conversationId,
              storageConversationId,
              userQuery,
              assistantAnswer: response.answer,
              artifacts: response.artifacts,
              isFirstMessage,
              previousSummary: activeSummary,
              onSaved: async (savedConversationId) => {
                activeStorageConversationIdRef.current = savedConversationId;
                setActiveConversationId((current) => (current === conversationId ? savedConversationId : current));
                const localMessages = readCache<ChatMessage[]>(cacheKey(authUser.uid, `messages:${conversationId}`));
                if (localMessages) {
                  writeCache(cacheKey(authUser.uid, `messages:${savedConversationId}`), localMessages);
                  removeCache(cacheKey(authUser.uid, `messages:${conversationId}`));
                }
                setConversations((current) =>
                  current.map((conversation) =>
                    conversation.id === conversationId ? { ...conversation, id: savedConversationId } : conversation
                  )
                );
                const nextConversations = await safeStorage(() => listConversations(authUser.uid), conversations);
                const [nextProfile, nextMemories, nextProjects] = await Promise.all([
                  safeStorage(() => getUserProfile(authUser.uid), profile),
                  safeStorage(() => listMemories(authUser.uid), memories),
                  safeStorage(() => listProjects(authUser.uid), projects)
                ]);
                setProfile(nextProfile);
                setMemories(nextMemories);
                setProjects(nextProjects);
                setConversations(nextConversations);
                writeCache(cacheKey(authUser.uid, "profile"), nextProfile);
                writeCache(cacheKey(authUser.uid, "memories"), nextMemories);
                writeCache(cacheKey(authUser.uid, "projects"), nextProjects);
                writeCache(cacheKey(authUser.uid, "conversations"), nextConversations);
              }
            });
          },
          onError: (message) => setError(message)
        }
      );
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Something went wrong";
      setError(message);
    } finally {
      setIsLoading(false);
      setStreamStatus("");
    }
  }

  if (isBooting) {
    return <main className="flex min-h-screen items-center justify-center bg-[#f7f7f8] text-sm text-gray-500">Loading...</main>;
  }

  if (!authUser) {
    return <LoginScreen />;
  }

  return (
    <main className="flex h-screen overflow-hidden bg-[#f7f7f8] text-[#1f2328]">
      {isMobileSidebarOpen && (
        <div className="fixed inset-0 z-40 bg-black/35 md:hidden" onClick={() => setIsMobileSidebarOpen(false)} />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex h-dvh w-[86vw] max-w-80 shrink-0 flex-col overflow-hidden border-r border-black/10 bg-white shadow-xl transition-transform duration-200 md:static md:z-auto md:h-screen md:w-72 md:max-w-none md:translate-x-0 md:shadow-none ${
          isMobileSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:flex`}
      >
        <SidebarContent
          authEmail={authUser.email ?? ""}
          profileName={profile?.name}
          conversations={conversations}
          activeConversationId={activeConversationId}
          isChatsLoading={isChatsLoading}
          loadingConversationId={loadingConversationId}
          onNewChat={handleNewChat}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
          onClose={() => setIsMobileSidebarOpen(false)}
        />
      </aside>

      <div className="flex h-screen min-w-0 flex-1 flex-col overflow-hidden">
        <header className="shrink-0 border-b border-black/10 bg-white/80 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
            <div className="flex min-w-0 items-center gap-3">
              <button
                type="button"
                onClick={() => setIsMobileSidebarOpen(true)}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-gray-300 bg-white text-lg md:hidden"
                aria-label="Open chat history"
              >
                ☰
              </button>
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-black text-sm font-semibold text-white">AI</div>
              <div className="min-w-0">
                <h1 className="truncate text-sm font-semibold">Research Chatbot</h1>
                <p className="truncate text-xs text-gray-500">Chat history, memory, research, files</p>
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

        <section className="mx-auto flex w-full max-w-4xl flex-1 flex-col overflow-y-auto px-3 sm:px-4">
          {loadingConversationId && messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <h2 className="text-2xl font-semibold tracking-tight">Loading this chat...</h2>
              <p className="mt-3 text-sm text-gray-500">Messages are fetched only when you open a conversation.</p>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <h2 className="px-3 text-2xl font-semibold tracking-tight sm:text-3xl">How can I help you today?</h2>
              <p className="mt-3 max-w-md text-sm text-gray-500">Your chats, profile details, memories, and project context are saved to your account.</p>
            </div>
          ) : (
            <div className="flex-1 space-y-6 py-8">
              {messages.map((message, index) => (
                <MessageBubble
                  key={`${index}-${message.role}`}
                  message={message}
                  copied={copiedMessageIndex === index}
                  onCopy={async () => {
                    await navigator.clipboard.writeText(message.content);
                    setCopiedMessageIndex(index);
                    window.setTimeout(() => setCopiedMessageIndex(null), 1500);
                  }}
                />
              ))}
              {isLoading && <AssistantTyping status={streamStatus} />}
              {error && <div className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</div>}
            </div>
          )}
        </section>

        <footer className="shrink-0 border-t border-black/10 bg-[#f7f7f8]/90 px-3 py-3 backdrop-blur sm:px-4 sm:py-4">
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
          <p className="mt-2 hidden text-center text-xs text-gray-400 sm:block">Research Chatbot can make mistakes. Check important outputs.</p>
          </form>
        </footer>
      </div>
    </main>
  );
}

function SidebarContent({
  authEmail,
  profileName,
  conversations,
  activeConversationId,
  isChatsLoading,
  loadingConversationId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onClose
}: {
  authEmail: string;
  profileName?: string;
  conversations: StoredConversation[];
  activeConversationId: string;
  isChatsLoading: boolean;
  loadingConversationId: string;
  onNewChat: () => void;
  onSelectConversation: (conversationId: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <div className="flex items-center gap-2 border-b border-black/10 p-3">
        <button
          type="button"
          onClick={onNewChat}
          className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-left text-sm font-medium hover:bg-gray-50"
        >
          + New chat
        </button>
        <button
          type="button"
          onClick={onClose}
          className="flex h-9 w-9 items-center justify-center rounded-lg border border-gray-300 text-sm md:hidden"
          aria-label="Close chat history"
        >
          x
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {isChatsLoading && (
          <p className="px-2 py-3 text-xs text-gray-500">{conversations.length ? "Refreshing chats..." : "Loading previous chats..."}</p>
        )}
        {!isChatsLoading && conversations.length === 0 ? (
          <p className="px-2 py-4 text-xs text-gray-500">Your saved chats will appear here.</p>
        ) : (
          conversations.map((conversation) => (
            <div key={conversation.id} className="group flex items-center gap-1">
              <button
                type="button"
                onClick={() => onSelectConversation(conversation.id)}
                className={`min-w-0 flex-1 rounded-lg px-3 py-2 text-left text-sm ${
                  activeConversationId === conversation.id ? "bg-gray-100 font-medium" : "hover:bg-gray-50"
                }`}
              >
                <span className="block truncate">
                  {loadingConversationId === conversation.id ? "Loading chat..." : conversation.title || "New chat"}
                </span>
              </button>
              <button
                type="button"
                onClick={() => onDeleteConversation(conversation.id)}
                className="rounded-md px-2 py-1 text-xs text-gray-500 hover:bg-red-50 hover:text-red-600 md:opacity-0 md:group-hover:opacity-100"
                aria-label="Delete chat"
              >
                x
              </button>
            </div>
          ))
        )}
      </div>
      <div className="border-t border-black/10 p-3 text-xs">
        <div className="mb-2 truncate font-medium">{profileName || authEmail}</div>
        <div className="flex gap-2">
          <Link href="/profile" onClick={onClose} className="rounded-md border border-gray-300 px-2 py-1 hover:bg-gray-50">
            Profile
          </Link>
          <button type="button" onClick={() => signOut(auth)} className="rounded-md border border-gray-300 px-2 py-1 hover:bg-gray-50">
            Sign out
          </button>
        </div>
      </div>
    </>
  );
}

function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      await authReady;
      await signInWithEmailAndPassword(auth, email, password);
    } catch (loginError) {
      console.error(loginError);
      setError(friendlyFirebaseError(loginError));
    } finally {
      setIsLoading(false);
    }
  }

  async function handleGoogleLogin() {
    setError("");
    setIsLoading(true);
    try {
      await authReady;
      const credential = await signInWithPopup(auth, new GoogleAuthProvider());
      void upsertUserProfile(credential.user).catch((profileError) => console.error(profileError));
    } catch (loginError) {
      console.error(loginError);
      setError(friendlyFirebaseError(loginError));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f7f8] px-4">
      <form onSubmit={handleLogin} className="w-full max-w-sm rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
        <h1 className="text-xl font-semibold">Sign in</h1>
        <p className="mt-1 text-sm text-gray-500">Use your account to load saved chats and memory.</p>
        <input
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          type="email"
          placeholder="Email"
          className="mt-5 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
        />
        <input
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          type="password"
          placeholder="Password"
          className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
        />
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        <button disabled={isLoading} type="submit" className="mt-4 w-full rounded-lg bg-black px-3 py-2 text-sm font-medium text-white disabled:bg-gray-400">
          {isLoading ? "Signing in..." : "Sign in"}
        </button>
        <button
          disabled={isLoading}
          type="button"
          onClick={handleGoogleLogin}
          className="mt-3 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-800 hover:bg-gray-50 disabled:text-gray-400"
        >
          Continue with Google
        </button>
        <Link href="/signup" className="mt-3 block text-center text-sm text-gray-600 hover:text-black">
          Create a new account
        </Link>
      </form>
    </main>
  );
}

async function safeStorage<T>(operation: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await withTimeout(operation(), 5000);
  } catch (error) {
    console.error(error);
    return fallback;
  }
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error("Firebase request timed out. Check Firestore rules/network.")), timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

function profileFromFirebaseUser(user: User): StoredUser {
  return {
    id: user.uid,
    email: user.email ?? "",
    name: user.displayName || user.email?.split("@")[0] || "User"
  };
}

function cacheKey(userId: string, key: string) {
  return `research-chatbot:${userId}:${key}`;
}

function readCache<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function writeCache<T>(key: string, value: T) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(error);
  }
}

function removeCache(key: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(key);
  } catch (error) {
    console.error(error);
  }
}

async function persistChatTurn(input: {
  userId: string;
  conversationId: string;
  storageConversationId: string;
  userQuery: string;
  assistantAnswer: string;
  artifacts: Artifact[];
  isFirstMessage: boolean;
  previousSummary: string;
  onSaved: (conversationId: string) => Promise<void>;
}) {
  try {
    const savedConversationId =
      input.storageConversationId ||
      (input.conversationId.startsWith("local-")
        ? await withTimeout(createConversation(input.userId, makeTitle(input.userQuery)), 5000)
        : input.conversationId);

    await withTimeout(addMessage(savedConversationId, "user", input.userQuery), 5000);
    await withTimeout(addMessage(savedConversationId, "assistant", input.assistantAnswer, input.artifacts), 5000);
    await withTimeout(updateConversationMeta(savedConversationId, {
      summary: makeConversationSummary(input.previousSummary, input.userQuery, input.assistantAnswer),
      ...(input.isFirstMessage ? { title: makeTitle(input.userQuery) } : {})
    }), 5000);

    for (const candidate of extractMemoryCandidates(input.userQuery)) {
      await safeStorage(() => saveExtractedMemory(input.userId, savedConversationId, candidate.text, candidate.category, candidate.importance), undefined);
    }
    const projectCandidate = extractProjectCandidate(input.userQuery);
    if (projectCandidate) {
      await safeStorage(() => saveProjectState(input.userId, projectCandidate), undefined);
    }

    await input.onSaved(savedConversationId);
  } catch (error) {
    console.error(error);
  }
}

function MessageBubble({
  message,
  copied,
  onCopy
}: {
  message: ChatMessage;
  copied: boolean;
  onCopy: () => Promise<void>;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[92%] rounded-3xl px-4 py-3 text-sm leading-7 sm:max-w-[82%] ${isUser ? "bg-black text-white" : "bg-transparent text-gray-900"}`}>
        <MarkdownMessage content={message.content} />
        {message.artifacts && message.artifacts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.artifacts.map((artifact) => (
              <ArtifactView key={artifact.url} artifact={artifact} />
            ))}
          </div>
        )}
        {!isUser && (
          <div className="mt-3">
            <button
              type="button"
              onClick={onCopy}
              className="rounded-full border border-gray-300 bg-white px-3 py-1 text-xs font-medium text-gray-700 shadow-sm hover:bg-gray-100"
            >
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function ArtifactView({ artifact }: { artifact: Artifact }) {
  const url = artifactUrl(artifact.url);
  const isImage = ["jpg", "jpeg", "png", "webp"].includes(artifact.type.toLowerCase());

  if (isImage) {
    return (
      <div className="w-full space-y-2">
        <img src={url} alt={artifact.name} className="max-h-[320px] max-w-full rounded-2xl border border-gray-200 shadow-sm sm:max-h-[420px]" />
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex rounded-full border border-gray-300 bg-white px-3 py-1 text-xs font-medium text-gray-900 hover:bg-gray-100"
        >
          Download {artifact.type.toUpperCase()}
        </a>
      </div>
    );
  }

  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="rounded-full border border-gray-300 bg-white px-3 py-1 text-xs font-medium text-gray-900 hover:bg-gray-100"
    >
      Download {artifact.type.toUpperCase()}
    </a>
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

function AssistantTyping({ status }: { status: string }) {
  return (
    <div className="flex justify-start">
      <div className="rounded-3xl bg-transparent px-4 py-3 text-sm text-gray-500">{status || "Thinking..."}</div>
    </div>
  );
}
