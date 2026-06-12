import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  writeBatch
} from "firebase/firestore";
import { updateProfile, type User } from "firebase/auth";
import { db } from "./firebase";
import type { Artifact } from "./api";

export type StoredUser = {
  id: string;
  email: string;
  name: string;
};

export type StoredConversation = {
  id: string;
  user_id: string;
  title: string;
  summary: string;
  updated_at?: unknown;
};

export type StoredMessage = {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  artifacts?: Artifact[];
  created_at?: unknown;
};

export type StoredMemory = {
  id: string;
  user_id: string;
  text: string;
  category: string;
  importance: number;
  source_conversation_id: string;
};

export type StoredProject = {
  id: string;
  user_id: string;
  name: string;
  summary: string;
  current_state: string;
  tech_stack: string;
  decisions: string;
  updated_at?: unknown;
};

export async function upsertUserProfile(user: User, name?: string) {
  const cleanName = (name ?? user.displayName ?? user.email?.split("@")[0] ?? "User").trim();
  await setDoc(
    doc(db, "users", user.uid),
    {
      id: user.uid,
      email: user.email ?? "",
      name: cleanName,
      updated_at: serverTimestamp()
    },
    { merge: true }
  );
  if (name && user.displayName !== cleanName) {
    await updateProfile(user, { displayName: cleanName });
  }
}

export async function getUserProfile(userId: string): Promise<StoredUser | null> {
  const snapshot = await getDoc(doc(db, "users", userId));
  if (!snapshot.exists()) return null;
  return snapshot.data() as StoredUser;
}

export async function createConversation(userId: string, title = "New chat") {
  const ref = await addDoc(collection(db, "users", userId, "chats"), {
    user_id: userId,
    title,
    summary: "",
    created_at: serverTimestamp(),
    updated_at: serverTimestamp()
  });
  return ref.id;
}

export async function listConversations(userId: string): Promise<StoredConversation[]> {
  const snapshot = await getDocs(collection(db, "users", userId, "chats"));
  return snapshot.docs
    .map((item) => ({ id: item.id, ...(item.data() as Omit<StoredConversation, "id">) }))
    .sort((a, b) => timestampMillis(b.updated_at) - timestampMillis(a.updated_at));
}

export async function getConversation(userId: string, conversationId: string): Promise<StoredConversation | null> {
  const snapshot = await getDoc(doc(db, "users", userId, "chats", conversationId));
  if (!snapshot.exists()) return null;
  return { id: snapshot.id, ...(snapshot.data() as Omit<StoredConversation, "id">) };
}

export async function updateConversationMeta(userId: string, conversationId: string, data: Partial<Pick<StoredConversation, "title" | "summary">>) {
  await updateDoc(doc(db, "users", userId, "chats", conversationId), {
    ...data,
    updated_at: serverTimestamp()
  });
}

export async function deleteConversation(userId: string, conversationId: string) {
  const messagesSnapshot = await getDocs(collection(db, "users", userId, "chats", conversationId, "messages"));
  const batch = writeBatch(db);
  messagesSnapshot.docs.forEach((messageDoc) => batch.delete(messageDoc.ref));
  batch.delete(doc(db, "users", userId, "chats", conversationId));
  await batch.commit();
}

export async function addMessage(
  userId: string,
  conversationId: string,
  role: StoredMessage["role"],
  content: string,
  artifacts?: Artifact[]
) {
  await addDoc(collection(db, "users", userId, "chats", conversationId, "messages"), {
    conversation_id: conversationId,
    role,
    content,
    artifacts: artifacts ?? [],
    created_at: serverTimestamp()
  });
}

export async function listMessages(userId: string, conversationId: string): Promise<StoredMessage[]> {
  const snapshot = await getDocs(collection(db, "users", userId, "chats", conversationId, "messages"));
  return snapshot.docs
    .map((item) => ({ id: item.id, ...(item.data() as Omit<StoredMessage, "id">) }))
    .sort((a, b) => timestampMillis(a.created_at) - timestampMillis(b.created_at));
}

export async function listMemories(userId: string): Promise<StoredMemory[]> {
  const snapshot = await getDocs(collection(db, "users", userId, "memories"));
  return snapshot.docs
    .map((item) => ({ id: item.id, ...(item.data() as Omit<StoredMemory, "id">) }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 20);
}

export async function listProjects(userId: string): Promise<StoredProject[]> {
  const snapshot = await getDocs(collection(db, "users", userId, "projects"));
  return snapshot.docs
    .map((item) => ({ id: item.id, ...(item.data() as Omit<StoredProject, "id">) }))
    .sort((a, b) => timestampMillis(b.updated_at) - timestampMillis(a.updated_at))
    .slice(0, 8);
}

export async function saveExtractedMemory(userId: string, conversationId: string, text: string, category: string, importance = 0.7) {
  const cleanText = text.trim();
  if (!cleanText) return;
  await addDoc(collection(db, "users", userId, "memories"), {
    user_id: userId,
    text: cleanText,
    category,
    importance,
    source_conversation_id: conversationId,
    created_at: serverTimestamp()
  });
}

export async function saveProjectState(userId: string, data: Omit<StoredProject, "id" | "user_id">) {
  await addDoc(collection(db, "users", userId, "projects"), {
    user_id: userId,
    ...data,
    created_at: serverTimestamp(),
    updated_at: serverTimestamp()
  });
}

export function makeTitle(text: string) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  return cleaned.length > 48 ? `${cleaned.slice(0, 48).trim()}...` : cleaned || "New chat";
}

export function makeConversationSummary(previous: string, userText: string, assistantText: string) {
  const next = [
    previous,
    `User: ${userText}`,
    `Assistant: ${assistantText.slice(0, 700)}`
  ]
    .filter(Boolean)
    .join("\n");
  return next.length > 3200 ? next.slice(next.length - 3200) : next;
}

export function extractMemoryCandidates(text: string) {
  const candidates: Array<{ text: string; category: string; importance: number }> = [];
  const nameMatch = text.match(/\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z]{1,40}(?:\s+[A-Z][a-zA-Z]{1,40})?)/i);
  if (nameMatch) {
    candidates.push({ text: `User name: ${nameMatch[1].trim()}`, category: "identity", importance: 1 });
  }
  const preferenceMatch = text.match(/\b(?:i prefer|i like|my preference is|remember that i prefer)\s+(.{4,180})/i);
  if (preferenceMatch) {
    candidates.push({ text: `User preference: ${preferenceMatch[1].trim()}`, category: "preference", importance: 0.85 });
  }
  const rememberMatch = text.match(/\bremember (?:that )?(.{4,220})/i);
  if (rememberMatch) {
    candidates.push({ text: rememberMatch[1].trim(), category: "explicit_memory", importance: 0.95 });
  }
  const projectMatch = text.match(/\b(?:project|app|repo|chatbot)\b.{0,180}/i);
  if (projectMatch) {
    candidates.push({ text: `Project context: ${projectMatch[0].trim()}`, category: "project", importance: 0.75 });
  }
  return candidates;
}

export function extractProjectCandidate(text: string): Omit<StoredProject, "id" | "user_id"> | null {
  const projectMatch = text.match(/\b(?:my|our|the)?\s*(project|app|repo|chatbot|website)\s+(?:is|called|named|uses|has|should|will|needs)\s+(.{8,260})/i);
  if (!projectMatch) return null;

  const summary = projectMatch[0].replace(/\s+/g, " ").trim();
  const techMatch = text.match(/\b(?:using|with|built in|built with|tech stack is)\s+(.{3,120})/i);
  const decisionMatch = text.match(/\b(?:decision|decided|we will|we should|must)\s+(.{4,160})/i);

  return {
    name: makeTitle(summary.replace(/^(my|our|the)\s+/i, "")),
    summary,
    current_state: "Captured from user conversation",
    tech_stack: techMatch?.[1]?.trim() ?? "Not specified",
    decisions: decisionMatch?.[1]?.trim() ?? "Not specified"
  };
}

export function buildLongTermContext(input: {
  user: StoredUser | null;
  memories: StoredMemory[];
  projects: StoredProject[];
  conversationSummary: string;
}) {
  const sections = [];
  if (input.user) {
    sections.push(`USER PROFILE:\n- id: ${input.user.id}\n- email: ${input.user.email}\n- name: ${input.user.name}`);
  }
  if (input.memories.length) {
    sections.push(
      `LONG TERM MEMORIES:\n${input.memories
        .slice(0, 12)
        .map((memory) => `- [${memory.category}, importance ${memory.importance}] ${memory.text}`)
        .join("\n")}`
    );
  }
  if (input.projects.length) {
    sections.push(
      `PROJECTS:\n${input.projects
        .slice(0, 5)
        .map(
          (project) =>
            `- ${project.name}: ${project.summary}; state: ${project.current_state}; tech: ${project.tech_stack}; decisions: ${project.decisions}`
        )
        .join("\n")}`
    );
  }
  if (input.conversationSummary) {
    sections.push(`CURRENT CONVERSATION SUMMARY:\n${input.conversationSummary}`);
  }
  return sections.join("\n\n").slice(-5000);
}

function timestampMillis(value: unknown) {
  if (value && typeof value === "object" && "toMillis" in value) {
    const timestamp = value as { toMillis?: () => number };
    if (typeof timestamp.toMillis === "function") return timestamp.toMillis();
  }
  return 0;
}
