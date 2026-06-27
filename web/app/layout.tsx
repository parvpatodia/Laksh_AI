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
  title: "Laksh.ai -- Sports Biomechanics Analysis",
  description:
    "Frame-by-frame biomechanical analysis for basketball and strength training. " +
    "33 skeletal landmarks, real-time pose overlay, and a verified per-rep report " +
    "measured from your own body, not estimated from population averages.",
  openGraph: {
    title: "Laksh.ai -- Sports Biomechanics Analysis",
    description:
      "Real-time skeleton overlay and a verified biomechanical breakdown for every " +
      "shot or rep -- grounded in pose measurement, not population averages.",
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

        {/* Global navigation */}
        <header className="sticky top-0 z-50 border-b border-surface-700/60
                           bg-surface-900/90 backdrop-blur-md">
          <div className="max-w-screen-2xl mx-auto px-6 xl:px-12 py-0 flex items-center justify-between h-14">

            {/* Logo */}
            <a href="/" className="flex items-center gap-2 group">
              <span className="text-xl font-black tracking-tight text-white group-hover:text-brand-400 transition-colors">
                LAKSH
              </span>
              <span className="text-xl font-black tracking-tight text-brand-500">.AI</span>
            </a>

            {/* Nav links */}
            <nav className="flex items-center gap-1">
              <a
                href="/basketball"
                className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-white
                           hover:bg-surface-700/60 rounded-lg transition-all duration-150"
              >
                Basketball
              </a>
              <a
                href="/gym"
                className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-white
                           hover:bg-surface-700/60 rounded-lg transition-all duration-150"
              >
                Gym
              </a>
              <a
                href="/leaderboard"
                className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-white
                           hover:bg-surface-700/60 rounded-lg transition-all duration-150"
              >
                Leaderboard
              </a>
              <a
                href="https://github.com/parvpatodia/Laksh_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 px-3 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-300
                           border border-surface-600 hover:border-surface-500 rounded-lg transition-all duration-150"
              >
                GitHub
              </a>
            </nav>
          </div>
        </header>

        <main className="flex-1">{children}</main>

      </body>
    </html>
  );
}
