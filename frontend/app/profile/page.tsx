"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { onAuthStateChanged, updateProfile, type User } from "firebase/auth";
import { auth } from "../../lib/firebase";
import { friendlyFirebaseError } from "../../lib/firebaseErrors";
import {
  getUserProfile,
  listMemories,
  listProjects,
  upsertUserProfile,
  type StoredMemory,
  type StoredProject,
  type StoredUser
} from "../../lib/chatStorage";

export default function ProfilePage() {
  const router = useRouter();
  const [authUser, setAuthUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<StoredUser | null>(null);
  const [name, setName] = useState("");
  const [memories, setMemories] = useState<StoredMemory[]>([]);
  const [projects, setProjects] = useState<StoredProject[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    return onAuthStateChanged(auth, async (user) => {
      if (!user) {
        router.push("/");
        return;
      }
      setAuthUser(user);
      const [nextProfile, nextMemories, nextProjects] = await Promise.all([
        safeStorage(() => getUserProfile(user.uid), null),
        safeStorage(() => listMemories(user.uid), []),
        safeStorage(() => listProjects(user.uid), [])
      ]);
      setProfile(nextProfile);
      setName(nextProfile?.name ?? user.displayName ?? "");
      setMemories(nextMemories);
      setProjects(nextProjects);
    });
  }, [router]);

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!authUser) return;
    setStatus("");
    setError("");
    setIsSaving(true);
    try {
      const cleanName = name.trim() || authUser.email?.split("@")[0] || "User";
      await withTimeout(updateProfile(authUser, { displayName: cleanName }), 5000);
      const fallbackProfile = {
        id: authUser.uid,
        email: authUser.email ?? "",
        name: cleanName
      };
      let firestoreSaved = true;
      try {
        await withTimeout(upsertUserProfile(authUser, cleanName), 5000);
      } catch (storageError) {
        firestoreSaved = false;
        console.error(storageError);
      }
      const nextProfile = await safeStorage(() => getUserProfile(authUser.uid), fallbackProfile);
      setProfile(nextProfile);
      setStatus(firestoreSaved ? "Profile saved." : "Name updated locally. Firestore blocked profile storage; check your Firestore rules.");
    } catch (saveError) {
      console.error(saveError);
      setError(friendlyFirebaseError(saveError));
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f7f8] px-4 py-8 text-[#1f2328]">
      <div className="mx-auto max-w-3xl">
        <Link href="/" className="text-sm text-gray-600 hover:text-black">
          ← Back to chat
        </Link>

        <section className="mt-5 rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
          <h1 className="text-xl font-semibold">Profile</h1>
          <p className="mt-1 text-sm text-gray-500">Update the account details used in long-term context.</p>
          <form onSubmit={handleSave} className="mt-5 space-y-3">
            <label className="block text-sm font-medium">
              Name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
              />
            </label>
            <label className="block text-sm font-medium">
              Email
              <input
                value={profile?.email ?? authUser?.email ?? ""}
                disabled
                className="mt-1 w-full rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-500 outline-none"
              />
            </label>
            {status && <p className="text-sm text-green-700">{status}</p>}
            {error && <p className="text-sm text-red-600">{error}</p>}
            <button disabled={isSaving} type="submit" className="rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-gray-400">
              {isSaving ? "Saving..." : "Save profile"}
            </button>
          </form>
        </section>

        <section className="mt-5 grid gap-5 md:grid-cols-2">
          <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold">Saved memories</h2>
            <div className="mt-3 space-y-3">
              {memories.length === 0 ? (
                <p className="text-sm text-gray-500">No memories saved yet.</p>
              ) : (
                memories.slice(0, 10).map((memory) => (
                  <div key={memory.id} className="rounded-lg border border-gray-100 p-3 text-sm">
                    <div className="text-xs font-medium uppercase tracking-wide text-gray-400">{memory.category}</div>
                    <p className="mt-1 text-gray-800">{memory.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold">Projects</h2>
            <div className="mt-3 space-y-3">
              {projects.length === 0 ? (
                <p className="text-sm text-gray-500">Project state will appear here as it is saved.</p>
              ) : (
                projects.slice(0, 8).map((project) => (
                  <div key={project.id} className="rounded-lg border border-gray-100 p-3 text-sm">
                    <div className="font-medium">{project.name}</div>
                    <p className="mt-1 text-gray-600">{project.summary}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </div>
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
