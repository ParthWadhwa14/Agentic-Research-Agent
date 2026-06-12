export function friendlyFirebaseError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("auth/configuration-not-found")) {
    return "Firebase Authentication is not enabled for this project. Open Firebase Console > Authentication > Get started, then enable Email/Password sign-in.";
  }
  if (message.includes("auth/operation-not-allowed")) {
    return "Email/password sign-in is disabled. Open Firebase Console > Authentication > Sign-in method and enable Email/Password.";
  }
  if (message.includes("auth/invalid-credential")) {
    return "The email or password is incorrect, or this account does not exist yet. Use Create a new account first.";
  }
  if (message.includes("auth/user-not-found")) {
    return "No account exists for this email yet. Use Create a new account first.";
  }
  if (message.includes("auth/wrong-password")) {
    return "The password is incorrect.";
  }
  if (message.includes("auth/email-already-in-use")) {
    return "An account already exists for this email. Go back and sign in instead.";
  }
  if (message.includes("auth/popup-closed-by-user")) {
    return "The Google sign-in popup was closed before finishing.";
  }
  if (message.includes("permission-denied") || message.includes("Missing or insufficient permissions")) {
    return "Firestore permissions blocked this action. Check your Firestore security rules for signed-in users.";
  }
  return message || "Something went wrong with Firebase.";
}
