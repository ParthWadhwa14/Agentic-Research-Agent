"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createUserWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from "firebase/auth";
import { auth, authReady } from "../../lib/firebase";
import { friendlyFirebaseError } from "../../lib/firebaseErrors";
import { upsertUserProfile } from "../../lib/chatStorage";

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSignup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      await authReady;
      const credential = await createUserWithEmailAndPassword(auth, email, password);
      void upsertUserProfile(credential.user, name).catch((profileError) => console.error(profileError));
      router.replace("/");
      window.setTimeout(() => {
        if (window.location.pathname !== "/") window.location.assign("/");
      }, 800);
    } catch (signupError) {
      console.error(signupError);
      setError(friendlyFirebaseError(signupError));
      setIsLoading(false);
    }
  }

  async function handleGoogleSignup() {
    setError("");
    setIsLoading(true);
    try {
      await authReady;
      const credential = await signInWithPopup(auth, new GoogleAuthProvider());
      void upsertUserProfile(credential.user, name || credential.user.displayName || undefined).catch((profileError) => console.error(profileError));
      router.replace("/");
      window.setTimeout(() => {
        if (window.location.pathname !== "/") window.location.assign("/");
      }, 800);
    } catch (signupError) {
      console.error(signupError);
      setError(friendlyFirebaseError(signupError));
      setIsLoading(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f7f8] px-4">
      <form onSubmit={handleSignup} className="w-full max-w-sm rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
        <h1 className="text-xl font-semibold">Create account</h1>
        <p className="mt-1 text-sm text-gray-500">Your chats and long-term memory will be saved to this account.</p>
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Name"
          className="mt-5 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
          required
        />
        <input
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          type="email"
          placeholder="Email"
          className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
          required
        />
        <input
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          type="password"
          placeholder="Password"
          className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none"
          minLength={6}
          required
        />
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        <button disabled={isLoading} type="submit" className="mt-4 w-full rounded-lg bg-black px-3 py-2 text-sm font-medium text-white disabled:bg-gray-400">
          {isLoading ? "Creating..." : "Sign up"}
        </button>
        <button
          disabled={isLoading}
          type="button"
          onClick={handleGoogleSignup}
          className="mt-3 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-800 hover:bg-gray-50 disabled:text-gray-400"
        >
          Continue with Google
        </button>
        <Link href="/" className="mt-3 block text-center text-sm text-gray-600 hover:text-black">
          Back to sign in
        </Link>
      </form>
    </main>
  );
}
