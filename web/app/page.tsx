import Link from "next/link";
import FailureModeCards from "@/components/FailureModeCards";

const SPORTS = [
  {
    id: "basketball",
    name: "Basketball",
    subtitle: "Jump shot biomechanics",
    description:
      "Point your camera and shoot. Get real-time skeleton overlay while you record, " +
      "then upload for a full breakdown of release speed, arc, and body mechanics.",
    icon: "🏀",
    href: "/basketball",
    available: true,
    metrics: [
      "Release speed",
      "Shot arc",
      "Knee bend",
      "Elbow angle",
      "Body sync timing",
      "Fluidity score",
    ],
  },
  {
    id: "gym",
    name: "Gym",
    subtitle: "Rep-by-rep lift analysis",
    description:
      "12 compound movements supported. See every rep scored on tempo, range of motion, " +
      "and consistency. Honest results — only what the camera can actually measure.",
    icon: "🏋️",
    href: "/gym",
    available: true,
    metrics: [
      "Rep duration",
      "Lowering phase",
      "Lifting phase",
      "Tempo ratio",
      "Range of motion",
      "Joint tracking quality",
    ],
  },
];

export default function HomePage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      {/* Hero */}
      <div className="mb-16 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-slate-100 mb-4">
          Sports biomechanics,{" "}
          <span className="text-brand-500">honest by construction</span>
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          Live skeleton overlay while you move. Upload your clip and get a full
          biomechanical breakdown — every metric backed by pose data from your own body,
          with honest gaps when the camera can&apos;t measure.
        </p>
      </div>

      {/* Sport cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
        {SPORTS.map((sport) => (
          <Link
            key={sport.id}
            href={sport.href}
            className="group block rounded-2xl border border-surface-700 bg-surface-800 p-6
                       hover:border-brand-500/60 hover:bg-surface-700/50 transition-all duration-200"
          >
            <div className="flex items-center gap-3 mb-3">
              <span className="text-3xl">{sport.icon}</span>
              <div>
                <h2 className="text-xl font-semibold text-slate-100 group-hover:text-brand-500 transition-colors">
                  {sport.name}
                </h2>
                <p className="text-sm text-slate-500">{sport.subtitle}</p>
              </div>
            </div>
            <p className="text-sm text-slate-400 mb-4 leading-relaxed">
              {sport.description}
            </p>
            <ul className="grid grid-cols-2 gap-1">
              {sport.metrics.map((m) => (
                <li
                  key={m}
                  className="text-xs text-slate-500 flex items-center gap-1"
                >
                  <span className="w-1 h-1 rounded-full bg-brand-500/60 inline-block" />
                  {m}
                </li>
              ))}
            </ul>
            <div className="mt-4 flex items-center text-sm text-brand-500 font-medium">
              Start analysis
              <svg
                className="ml-1 w-4 h-4 group-hover:translate-x-1 transition-transform"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </div>
          </Link>
        ))}
      </div>

      {/* Research contribution callout */}
      <div className="rounded-xl border border-surface-700 bg-surface-800/50 p-6">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">
          What makes this different
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-slate-400">
          <div>
            <p className="font-medium text-slate-200 mb-1">No invented numbers</p>
            <p>
              Every metric reports what was actually measured and why.
              If a joint was out of frame, that metric shows blank — not a guess.
            </p>
          </div>
          <div>
            <p className="font-medium text-slate-200 mb-1">Honest reference ranges</p>
            <p>
              Metrics are only graded against reference data when that data exists.
              Until then they&apos;re shown raw so you can still compare across sessions.
            </p>
          </div>
          <div>
            <p className="font-medium text-slate-200 mb-1">Live vs. video cross-check</p>
            <p>
              The live counter and the video analysis run independently.
              Any difference between them is surfaced so you can see how accurate the live estimate was.
            </p>
          </div>
        </div>
      </div>

      {/* Failure modes & honesty section */}
      <FailureModeCards />
    </div>
  );
}
