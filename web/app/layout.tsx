import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Laksh.ai — Sports Biomechanics Analysis",
  description:
    "Frame-by-frame biomechanical analysis for basketball and strength training. " +
    "33 skeletal landmarks, real-time pose overlay, and a verified per-rep report — " +
    "measured from your own body, not estimated from population averages.",
  openGraph: {
    title: "Laksh.ai — Sports Biomechanics Analysis",
    description:
      "Real-time skeleton overlay and a verified biomechanical breakdown for every " +
      "shot or rep — grounded in pose measurement, not population averages.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`h-full ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="h-full antialiased bg-surface-900 text-slate-200 font-sans">
        <header className="border-b border-surface-700 px-6 py-4 flex items-center justify-between">
          <a href="/" className="text-lg font-semibold tracking-tight">
            <span className="text-brand-500">Laksh</span>
            <span className="text-slate-400">.ai</span>
          </a>
          <nav className="flex items-center gap-6 text-sm text-slate-400">
            <a href="/basketball" className="hover:text-slate-200 transition-colors">
              Basketball
            </a>
            <a href="/gym" className="hover:text-slate-200 transition-colors">
              Gym
            </a>
            <a
              href="https://github.com/parvpatodia/Laksh_AI"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-slate-200 transition-colors"
            >
              GitHub
            </a>
          </nav>
        </header>
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
