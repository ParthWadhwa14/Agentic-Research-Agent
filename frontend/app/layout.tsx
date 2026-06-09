import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Research Agentic Chatbot",
  description: "Agentic chat, deep research, file analysis, reports, and PPT generation"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
