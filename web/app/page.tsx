import Link from "next/link";

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const SPORTS = [
  {
    id: "basketball",
    href: "/basketball",
    name: "Basketball",
    subtitle: "Jump shot mechanics",
    description:
      "Record yourself shooting and get a frame-by-frame breakdown of every " +
      "mechanical variable that determines whether the ball goes in. Measured " +
      "from your own body — not estimated from a population average.",
    metrics: [
      { label: "Release speed", unit: "m/s", sample: "6.8" },
      { label: "Shot arc", unit: "°", sample: "48" },
      { label: "Knee bend at release", unit: "°", sample: "138" },
      { label: "Elbow angle", unit: "°", sample: "162" },
      { label: "Body sync timing", unit: "ms", sample: "290" },
      { label: "Fluidity score", unit: "/100", sample: "71" },
    ],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6 text-brand-500" stroke="currentColor" strokeWidth={1.5}>
        <circle cx="12" cy="12" r="9" />
        <path strokeLinecap="round" d="M3.6 9h16.8M3.6 15h16.8" />
        <path strokeLinecap="round" d="M12 3a15 15 0 0 1 4 9 15 15 0 0 1-4 9M12 3a15 15 0 0 0-4 9 15 15 0 0 0 4 9" />
      </svg>
    ),
  },
  {
    id: "gym",
    href: "/gym",
    name: "Strength Training",
    subtitle: "Rep-by-rep movement audit",
    description:
      "12 compound lifts supported. Every rep is scored independently on tempo, " +
      "range of motion, and joint tracking quality. Partial reps are flagged — " +
      "not silently counted as full reps.",
    metrics: [
      { label: "Rep duration", unit: "s", sample: "2.4" },
      { label: "Lowering phase", unit: "s", sample: "1.6" },
      { label: "Lifting phase", unit: "s", sample: "0.8" },
      { label: "Tempo ratio", unit: "×", sample: "2.0" },
      { label: "Range of motion", unit: "°", sample: "118" },
      { label: "Joint tracking quality", unit: "%", sample: "91" },
    ],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6 text-brand-500" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 8h2m8 0h2M8 8V6a1 1 0 0 1 1-1h1m4 0h1a1 1 0 0 1 1 1v2M8 8h8m0 0v8M8 8v8m0 0H6a1 1 0 0 1-1-1v-1m11 2h2a1 1 0 0 0 1-1v-1" />
        <rect x="9" y="8" width="6" height="8" rx="0.5" />
      </svg>
    ),
  },
];

const PIPELINE_STEPS = [
  {
    step: "01",
    title: "Live pose tracking",
    body:
      "33 skeletal landmarks tracked in your browser at 30 fps using MediaPipe Pose. " +
      "Skeleton overlay renders in real time — no footage leaves the device until you upload.",
  },
  {
    step: "02",
    title: "Multi-signal analysis",
    body:
      "The uploaded clip is processed with a server-side MediaPipe Heavy model. " +
      "Release events are confirmed by two independent kinematic signals — wrist trajectory " +
      "and elbow angular velocity — so a single noisy frame cannot fabricate a detection.",
  },
  {
    step: "03",
    title: "Verified biomech report",
    body:
      "Each metric is returned with its measurement confidence and the specific frames used. " +
      "Fields that could not be reliably measured are left blank with an explanation — never " +
      "filled with a population average or interpolated guess.",
  },
];

const METHODOLOGY_STATS = [
  { value: "33", label: "Skeletal landmarks" },
  { value: "2-of-N", label: "Signal consensus" },
  { value: "<40 ms", label: "Live latency" },
  { value: "12", label: "Lifts supported" },
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricPreviewRow({
  label,
  unit,
  sample,
}: {
  label: string;
  unit: string;
  sample: string;
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-surface-700/50 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-xs font-mono text-slate-200 tabular-nums">
        {sample}
        <span className="text-slate-500 ml-0.5">{unit}</span>
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  return (
    <div className="max-w-6xl mx-auto px-6">

      {/* ------------------------------------------------------------------ */}
      {/* Hero                                                                 */}
      {/* ------------------------------------------------------------------ */}
      <section className="pt-20 pb-16 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        {/* Left: headline + CTA */}
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-brand-500/30
                          bg-brand-500/10 text-brand-500 text-xs font-medium px-3 py-1.5 mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
            Sports biomechanics research
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-100 leading-tight mb-5">
            Movement analysis
            <br />
            <span className="text-brand-500">grounded in measurement</span>
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed mb-8 max-w-xl">
            Real-time skeleton overlay while you move. Upload your clip for a full
            biomechanical audit — every number traced back to the exact frame and
            joint it came from.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/basketball"
              className="inline-flex items-center gap-2 rounded-lg bg-brand-500 hover:bg-brand-600
                         text-white font-medium text-sm px-5 py-2.5 transition-colors"
            >
              Analyze a jump shot
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            <Link
              href="/gym"
              className="inline-flex items-center gap-2 rounded-lg border border-surface-700
                         hover:border-surface-600 text-slate-300 font-medium text-sm px-5 py-2.5 transition-colors"
            >
              Analyze a lift
            </Link>
          </div>
        </div>

        {/* Right: static sample report card */}
        <div className="rounded-2xl border border-surface-700 bg-surface-800 p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-xs text-slate-500 mb-0.5">Jump shot — sample output</p>
              <p className="text-sm font-semibold text-slate-200">Shot analysis</p>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-900/40
                             border border-emerald-700/50 text-emerald-400 text-xs font-medium px-2.5 py-1">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 8 8">
                <path d="M2.3 6.73L.6 4.53c-.4-1.04.46-1.73 1.1-.8l1.1 1.4 3.4-3.8c.6-.61 1.93-.31 1.7.7l-4.3 5.7c-.5.5-1.2.5-1.3.13z" />
              </svg>
              Verified
            </span>
          </div>

          {/* Metric rows */}
          <div className="mb-4">
            {SPORTS[0].metrics.map((m) => (
              <MetricPreviewRow key={m.label} {...m} />
            ))}
          </div>

          {/* Confidence bar */}
          <div className="rounded-lg bg-surface-900/60 p-3">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-slate-500">Measurement confidence</span>
              <span className="font-mono text-slate-300">84%</span>
            </div>
            <div className="h-1.5 rounded-full bg-surface-700 overflow-hidden">
              <div className="h-full rounded-full bg-brand-500" style={{ width: "84%" }} />
            </div>
            <p className="text-[10px] text-slate-600 mt-2">
              3 valid shots · 0 degraded · 0 dropped
            </p>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Methodology stats strip                                              */}
      {/* ------------------------------------------------------------------ */}
      <section className="border-y border-surface-700/60 py-8 mb-16">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          {METHODOLOGY_STATS.map(({ value, label }) => (
            <div key={label} className="text-center">
              <p className="text-2xl font-bold text-slate-100 font-mono mb-1">{value}</p>
              <p className="text-xs text-slate-500">{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* How it works                                                         */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-20">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-8">
          How it works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-0">
          {PIPELINE_STEPS.map((step, i) => (
            <div
              key={step.step}
              className={`relative p-6 ${i < PIPELINE_STEPS.length - 1
                ? "md:border-r border-surface-700/60"
                : ""}`}
            >
              {/* connector line on mobile */}
              {i < PIPELINE_STEPS.length - 1 && (
                <div className="md:hidden absolute bottom-0 left-6 w-px h-6 bg-surface-700/60" />
              )}
              <div className="text-xs font-mono text-brand-500 mb-3 opacity-70">{step.step}</div>
              <h3 className="text-sm font-semibold text-slate-200 mb-2">{step.title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Sport modules                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-20">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-8">
          Analysis modules
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {SPORTS.map((sport) => (
            <Link
              key={sport.id}
              href={sport.href}
              className="group rounded-2xl border border-surface-700 bg-surface-800 p-6
                         hover:border-brand-500/50 hover:bg-surface-700/40 transition-all duration-200"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-surface-900 border border-surface-700 p-2">
                    {sport.icon}
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-slate-100 group-hover:text-brand-500 transition-colors">
                      {sport.name}
                    </h3>
                    <p className="text-xs text-slate-500">{sport.subtitle}</p>
                  </div>
                </div>
                <svg
                  className="w-4 h-4 text-slate-600 group-hover:text-brand-500 group-hover:translate-x-0.5
                             transition-all duration-200 mt-1 shrink-0"
                  fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </div>

              <p className="text-sm text-slate-400 leading-relaxed mb-5">{sport.description}</p>

              {/* Metric preview table */}
              <div className="rounded-lg bg-surface-900/60 border border-surface-700/50 px-4 py-3">
                <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">
                  Measured outputs
                </p>
                {sport.metrics.map((m) => (
                  <MetricPreviewRow key={m.label} {...m} />
                ))}
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Measurement philosophy                                               */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-20">
        <div className="rounded-2xl border border-surface-700 bg-surface-800/40 p-8">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-6">
            Measurement philosophy
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
            <div>
              <div className="w-8 h-0.5 bg-brand-500 mb-4" />
              <p className="text-sm font-semibold text-slate-200 mb-2">
                No fabricated values
              </p>
              <p className="text-sm text-slate-500 leading-relaxed">
                Every output is computed from the pose data in your specific clip.
                If a joint leaves the frame, that metric is left blank — not
                substituted with a population average.
              </p>
            </div>
            <div>
              <div className="w-8 h-0.5 bg-brand-500 mb-4" />
              <p className="text-sm font-semibold text-slate-200 mb-2">
                Graded only where reference data exists
              </p>
              <p className="text-sm text-slate-500 leading-relaxed">
                Metrics are scored against published biomechanical literature only
                when that reference exists. Otherwise values are shown raw — useful
                for tracking personal progress across sessions.
              </p>
            </div>
            <div>
              <div className="w-8 h-0.5 bg-brand-500 mb-4" />
              <p className="text-sm font-semibold text-slate-200 mb-2">
                Live vs. video reconciliation
              </p>
              <p className="text-sm text-slate-500 leading-relaxed">
                The browser counter and the server analysis run independently on
                different data. When they disagree, both counts are shown — no
                silent reconciliation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Honest limits                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section className="mb-20">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">
          Honest limits
        </h2>
        <p className="text-sm text-slate-500 mb-6 max-w-2xl">
          When the system cannot reliably measure something, it says so — with the
          specific reason. The three most common cases are shown below.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <LimitCard
            title="Joint out of frame"
            badge="low visibility"
            badgeColor="amber"
            body="If a key landmark drops below the confidence threshold, the
                  affected metric is flagged and left blank. Repositioning the
                  camera until the full body is visible resolves this."
          />
          <LimitCard
            title="No reps detected"
            badge="no signal"
            badgeColor="slate"
            body="When the joint angle signal does not show enough variation to
                  locate rep boundaries, zero reps are reported. This prevents
                  noise bursts from being counted as movement."
          />
          <LimitCard
            title="Multiple people in frame"
            badge="ambiguous"
            badgeColor="amber"
            body="If two overlapping pose skeletons are detected, the system
                  cannot safely assign landmarks to one athlete and flags all
                  reps as unreliable. Shooting solo resolves this immediately."
          />
        </div>
      </section>

    </div>
  );
}

// ---------------------------------------------------------------------------
// Limit card (replaces FailureModeCards on homepage)
// ---------------------------------------------------------------------------

function LimitCard({
  title,
  badge,
  badgeColor,
  body,
}: {
  title: string;
  badge: string;
  badgeColor: "amber" | "slate" | "red";
  body: string;
}) {
  const badgeCls =
    badgeColor === "amber"
      ? "bg-amber-900/40 border-amber-700/50 text-amber-400"
      : badgeColor === "red"
      ? "bg-red-900/40 border-red-700/50 text-red-400"
      : "bg-slate-800 border-slate-700 text-slate-400";

  return (
    <div className="rounded-xl border border-surface-700 bg-surface-800 p-5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <p className="text-sm font-semibold text-slate-200">{title}</p>
        <span className={`shrink-0 text-[10px] font-medium border rounded px-1.5 py-0.5 ${badgeCls}`}>
          {badge}
        </span>
      </div>
      <p className="text-sm text-slate-500 leading-relaxed">{body}</p>
    </div>
  );
}
