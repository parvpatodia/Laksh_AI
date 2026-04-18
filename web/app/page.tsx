import Link from "next/link";

const SPORTS = [
  {
    id: "basketball",
    name: "Basketball",
    subtitle: "Jump shot biomechanics",
    description:
      "Release velocity, shot arc, knee/elbow flexion, kinetic chain sync. " +
      "Pose landmarker lite in-browser; heavy model on backend.",
    icon: "🏀",
    href: "/basketball",
    available: true,
    metrics: [
      "Release velocity (m/s)",
      "Shot arc (deg)",
      "Knee flexion (deg)",
      "Elbow flexion (deg)",
      "Kinetic sync (ms)",
      "Fluidity score (/100)",
    ],
  },
  {
    id: "gym",
    name: "Gym",
    subtitle: "Compound-lift rep analysis",
    description:
      "12 compound movements. Per-rep feature vector: duration, tempo ratio, " +
      "eccentric/concentric split, signal amplitude, visibility. " +
      "Honest calibration: uncalibrated_v0 until reference data lands.",
    icon: "🏋️",
    href: "/gym",
    available: true,
    metrics: [
      "Rep duration (s)",
      "Eccentric phase (s)",
      "Concentric phase (s)",
      "Tempo ratio",
      "Signal amplitude",
      "Min visibility",
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
          Real-time pose overlay in the browser. Canonical backend analysis with
          per-field{" "}
          <code className="font-mono text-sm bg-surface-800 px-1.5 py-0.5 rounded">
            status
          </code>{" "}
          +{" "}
          <code className="font-mono text-sm bg-surface-800 px-1.5 py-0.5 rounded">
            reason_codes
          </code>
          . Numeric parity probe between the two paths.
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
          Research contribution
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm text-slate-400">
          <div>
            <p className="font-medium text-slate-200 mb-1">Measurement spine</p>
            <p>
              Every number carries{" "}
              <code className="font-mono text-xs bg-surface-700 px-1 rounded">
                value, unit, status, reason_codes
              </code>
              . No bare floats.
            </p>
          </div>
          <div>
            <p className="font-medium text-slate-200 mb-1">Calibration honesty</p>
            <p>
              <code className="font-mono text-xs bg-surface-700 px-1 rounded">
                uncalibrated_v0
              </code>{" "}
              entries cannot claim reference ranges. Policy enforced at serialisation time.
            </p>
          </div>
          <div>
            <p className="font-medium text-slate-200 mb-1">Parity probe</p>
            <p>
              p90 absolute delta between browser ghost metrics and canonical
              backend result, reported per clip. Numerically auditable.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
