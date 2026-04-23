import Link from "next/link";

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const SPORTS = [
  {
    id: "basketball",
    href: "/basketball",
    name: "Basketball",
    tag: "Jump Shot Mechanics",
    description:
      "Record your jump shot and get a frame-by-frame breakdown of every " +
      "mechanical variable that determines whether the ball goes in. Measured " +
      "from your own body, not estimated from a population average.",
    metrics: [
      { label: "Release speed",        unit: "m/s",  sample: "6.8", tier: "good" },
      { label: "Shot arc",             unit: "deg",  sample: "48",  tier: "elite" },
      { label: "Knee bend at release", unit: "deg",  sample: "138", tier: "good" },
      { label: "Elbow angle",          unit: "deg",  sample: "162", tier: "elite" },
      { label: "Body sync timing",     unit: "ms",   sample: "290", tier: "good" },
      { label: "Fluidity score",       unit: "/100", sample: "71",  tier: "good" },
    ],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth={1.5}>
        <circle cx="12" cy="12" r="9" />
        <path strokeLinecap="round" d="M3.6 9h16.8M3.6 15h16.8" />
        <path strokeLinecap="round" d="M12 3a15 15 0 0 1 4 9 15 15 0 0 1-4 9M12 3a15 15 0 0 0-4 9 15 15 0 0 0 4 9" />
      </svg>
    ),
    accentColor: "from-brand-500/20 to-transparent",
    borderHover: "hover:border-brand-500/50",
  },
  {
    id: "gym",
    href: "/gym",
    name: "Strength Training",
    tag: "Rep-by-Rep Movement Audit",
    description:
      "12 compound lifts supported. Every rep is scored independently on tempo, " +
      "range of motion, and joint tracking quality. Partial reps are flagged, " +
      "not silently counted as full reps.",
    metrics: [
      { label: "Rep duration",          unit: "s",   sample: "2.4", tier: "good" },
      { label: "Lowering phase",        unit: "s",   sample: "1.6", tier: "elite" },
      { label: "Lifting phase",         unit: "s",   sample: "0.8", tier: "good" },
      { label: "Tempo ratio",           unit: "x",   sample: "2.0", tier: "elite" },
      { label: "Range of motion",       unit: "deg", sample: "118", tier: "good" },
      { label: "Joint tracking quality",unit: "%",   sample: "91",  tier: "elite" },
    ],
    icon: (
      <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 8h2m8 0h2M8 8V6a1 1 0 0 1 1-1h1m4 0h1a1 1 0 0 1 1 1v2M8 8h8m0 0v8M8 8v8m0 0H6a1 1 0 0 1-1-1v-1m11 2h2a1 1 0 0 0 1-1v-1" />
        <rect x="9" y="8" width="6" height="8" rx="0.5" />
      </svg>
    ),
    accentColor: "from-perf-500/15 to-transparent",
    borderHover: "hover:border-perf-500/40",
  },
];

const PIPELINE_STEPS = [
  {
    step: "01",
    title: "Live skeleton tracking",
    body:
      "33 anatomical landmarks tracked in your browser at 30 fps using MediaPipe Pose Lite. " +
      "Skeleton overlay renders in real time. No footage leaves the device until you choose to upload.",
  },
  {
    step: "02",
    title: "Multi-signal consensus",
    body:
      "The uploaded clip is processed server-side with MediaPipe Heavy. " +
      "Release events are confirmed by two independent kinematic signals -- " +
      "wrist trajectory and elbow angular velocity -- so a single noisy frame cannot fabricate a detection.",
  },
  {
    step: "03",
    title: "Verified biomech report",
    body:
      "Each metric is returned with its measurement source and confidence. " +
      "Fields that could not be reliably measured are left blank with an explanation -- " +
      "never filled with a population average or interpolated guess.",
  },
];

const METHODOLOGY_STATS = [
  { value: "33",     label: "Skeletal landmarks" },
  { value: "30 fps", label: "Live tracking rate" },
  { value: "80s",    label: "Analysis time" },
  { value: "12",     label: "Exercises supported" },
];

const PRINCIPLES = [
  {
    title: "No fabricated values",
    body:
      "Every output is computed from pose data in your specific clip. " +
      "If a joint leaves the frame, that metric is left blank -- not substituted with a population average.",
  },
  {
    title: "Graded where reference data exists",
    body:
      "Metrics are scored against published biomechanical literature only when that reference " +
      "exists. Otherwise values are shown raw, useful for tracking personal progress across sessions.",
  },
  {
    title: "Live vs. video reconciliation",
    body:
      "The browser counter and server analysis run independently on different data. " +
      "When they disagree, both counts are shown -- no silent reconciliation.",
  },
];

const LIMITS = [
  {
    title: "Joint out of frame",
    badge: "low visibility",
    body:
      "If a key landmark drops below the confidence threshold, the affected metric is " +
      "flagged and left blank. Repositioning the camera until the full body is visible resolves this.",
  },
  {
    title: "No reps detected",
    badge: "no signal",
    body:
      "When the joint angle signal does not show enough variation to locate rep boundaries, " +
      "zero reps are reported. This prevents noise bursts from being counted as movement.",
  },
  {
    title: "Multiple people in frame",
    badge: "ambiguous",
    body:
      "If two overlapping skeletons are detected, the system cannot safely assign landmarks " +
      "to one athlete and flags all reps as unreliable. Shooting solo resolves this immediately.",
  },
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

type Tier = "elite" | "good" | "none";

function TierBadge({ tier }: { tier: Tier }) {
  if (tier === "elite") {
    return (
      <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded
                       bg-emerald-900/50 text-emerald-400 border border-emerald-700/50">
        ELITE
      </span>
    );
  }
  if (tier === "good") {
    return (
      <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded
                       bg-sky-900/50 text-sky-400 border border-sky-700/50">
        GOOD
      </span>
    );
  }
  return null;
}

function MetricRow({
  label,
  unit,
  sample,
  tier,
}: {
  label: string;
  unit: string;
  sample: string;
  tier: Tier;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-surface-700/40 last:border-0 gap-3">
      <span className="text-xs text-slate-400 truncate">{label}</span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-sm font-bold font-mono text-slate-100 tabular-nums">
          {sample}
          <span className="text-slate-500 text-xs font-normal ml-0.5">{unit}</span>
        </span>
        <TierBadge tier={tier as Tier} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  return (
    <div className="bg-surface-900 min-h-screen">

      {/* ================================================================== */}
      {/* HERO                                                                */}
      {/* ================================================================== */}
      <section className="relative overflow-hidden">
        {/* Deep background gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-surface-950 via-surface-900 to-surface-900" />
        {/* Subtle radial accent -- brand color glow top-left */}
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full
                        bg-brand-500/5 blur-3xl pointer-events-none" />
        {/* Grid texture overlay */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,.15) 1px, transparent 1px), " +
              "linear-gradient(90deg, rgba(255,255,255,.15) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        <div className="relative z-10 max-w-screen-2xl mx-auto px-6 xl:px-16
                        pt-20 pb-16 lg:pt-32 lg:pb-24">
          <div className="max-w-4xl">

            {/* Eyebrow label */}
            <div className="inline-flex items-center gap-2 mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
              <span className="text-[11px] font-semibold uppercase tracking-[0.25em] text-brand-400">
                Sports Biomechanics Research
              </span>
            </div>

            {/* Headline */}
            <h1 className="headline-display text-5xl sm:text-6xl lg:text-7xl xl:text-8xl
                           font-black text-white mb-8">
              Read your<br />
              body<br />
              <span className="text-brand-500">in motion.</span>
            </h1>

            {/* Sub-headline */}
            <p className="text-lg xl:text-xl text-slate-400 leading-relaxed max-w-2xl mb-12">
              33-point skeletal tracking. Real measurements derived from your camera.
              Personalised coaching generated from what your body actually does -- not
              from population averages.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap gap-4">
              <Link
                href="/basketball"
                className="inline-flex items-center gap-2.5 rounded-xl bg-brand-500 hover:bg-brand-400
                           text-white font-semibold text-sm px-7 py-3.5 transition-all duration-200
                           shadow-lg shadow-brand-500/30 hover:shadow-brand-500/50 hover:-translate-y-0.5"
              >
                Analyse a jump shot
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </Link>
              <Link
                href="/gym"
                className="inline-flex items-center gap-2.5 rounded-xl border border-surface-600
                           hover:border-slate-500 bg-surface-800/60 hover:bg-surface-750
                           text-slate-300 hover:text-white font-semibold text-sm px-7 py-3.5
                           transition-all duration-200 hover:-translate-y-0.5"
              >
                Analyse a lift
              </Link>
            </div>
          </div>
        </div>

        {/* Stats bar */}
        <div className="relative z-10 border-t border-surface-700/60 bg-surface-850/60 backdrop-blur-sm">
          <div className="max-w-screen-2xl mx-auto px-6 xl:px-16 py-6">
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-surface-700/50">
              {METHODOLOGY_STATS.map(({ value, label }) => (
                <div key={label} className="text-center px-4 sm:px-8 first:pl-0 last:pr-0">
                  <p className="stat-number text-2xl xl:text-3xl text-white mb-1">{value}</p>
                  <p className="text-[10px] font-medium uppercase tracking-widest text-slate-500">{label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* SPORT MODULES                                                       */}
      {/* ================================================================== */}
      <section className="py-20 xl:py-28">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16">

          <div className="flex items-center gap-4 mb-12">
            <span className="label-section">Analysis modules</span>
            <div className="flex-1 h-px bg-surface-700/50" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 xl:gap-6">
            {SPORTS.map((sport) => (
              <Link
                key={sport.id}
                href={sport.href}
                className={`sport-card group block rounded-2xl border border-surface-700
                            bg-surface-800 overflow-hidden ${sport.borderHover}`}
              >
                {/* Gradient accent at top */}
                <div className={`h-0.5 w-full bg-gradient-to-r ${sport.accentColor}`} />

                <div className="p-7 xl:p-8">
                  {/* Sport header */}
                  <div className="flex items-start justify-between mb-5">
                    <div className="flex items-center gap-4">
                      <div className="rounded-xl bg-surface-900 border border-surface-700/60 p-3 text-brand-500">
                        {sport.icon}
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-white mb-0.5
                                       group-hover:text-brand-400 transition-colors">
                          {sport.name}
                        </h3>
                        <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
                          {sport.tag}
                        </p>
                      </div>
                    </div>
                    <div className="rounded-full p-2 bg-surface-700/50 text-slate-600
                                    group-hover:text-brand-500 group-hover:bg-brand-500/10
                                    transition-all duration-200 mt-1 shrink-0">
                      <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform"
                           fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </div>

                  <p className="text-sm text-slate-400 leading-relaxed mb-6">{sport.description}</p>

                  {/* Metric table */}
                  <div className="rounded-xl bg-surface-900/80 border border-surface-700/40 px-5 py-1">
                    <p className="label-section pt-3 pb-1">Measured outputs</p>
                    {sport.metrics.map((m) => (
                      <MetricRow key={m.label} label={m.label} unit={m.unit} sample={m.sample} tier={m.tier as Tier} />
                    ))}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* HOW IT WORKS                                                        */}
      {/* ================================================================== */}
      <section className="py-20 xl:py-24 border-t border-surface-700/40">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16">

          <div className="flex items-center gap-4 mb-14">
            <span className="label-section">How it works</span>
            <div className="flex-1 h-px bg-surface-700/50" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-0">
            {PIPELINE_STEPS.map((step, i) => (
              <div
                key={step.step}
                className={`relative py-8 px-8 xl:px-10
                  ${i < PIPELINE_STEPS.length - 1 ? "md:border-r border-surface-700/50" : ""}`}
              >
                <div className="text-5xl xl:text-6xl font-black font-mono text-surface-700
                                select-none mb-5 leading-none">
                  {step.step}
                </div>
                <h3 className="text-base font-bold text-white mb-3">{step.title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* MEASUREMENT PRINCIPLES                                              */}
      {/* ================================================================== */}
      <section className="py-20 xl:py-24 border-t border-surface-700/40">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16">

          <div className="flex items-center gap-4 mb-14">
            <span className="label-section">Measurement principles</span>
            <div className="flex-1 h-px bg-surface-700/50" />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 xl:gap-12">
            {PRINCIPLES.map((p) => (
              <div key={p.title}>
                <span className="rule-brand" />
                <h3 className="text-base font-bold text-white mb-3">{p.title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed">{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* HONEST LIMITS                                                       */}
      {/* ================================================================== */}
      <section className="py-20 xl:py-24 border-t border-surface-700/40">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16">

          <div className="flex items-center gap-4 mb-4">
            <span className="label-section">Measurement limits</span>
            <div className="flex-1 h-px bg-surface-700/50" />
          </div>
          <p className="text-sm text-slate-500 mb-10 max-w-2xl">
            When the system cannot reliably measure something, it reports exactly that --
            with the specific reason and the value that failed the threshold.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {LIMITS.map((limit) => (
              <div
                key={limit.title}
                className="rounded-2xl border border-surface-700/60 bg-surface-800/60 p-6"
              >
                <div className="flex items-start justify-between gap-3 mb-4">
                  <h3 className="text-sm font-bold text-white">{limit.title}</h3>
                  <span className="shrink-0 text-[9px] font-semibold uppercase tracking-wider
                                   border border-amber-700/50 bg-amber-900/30 text-amber-400
                                   rounded px-2 py-0.5 mt-0.5">
                    {limit.badge}
                  </span>
                </div>
                <p className="text-sm text-slate-500 leading-relaxed">{limit.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* FOOTER                                                              */}
      {/* ================================================================== */}
      <footer className="border-t border-surface-700/40 py-10">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-black text-white">LAKSH</span>
            <span className="text-sm font-black text-brand-500">.AI</span>
            <span className="text-xs text-slate-600 ml-2">Sports Biomechanics Research</span>
          </div>
          <p className="text-xs text-slate-600">
            Measurements derived from pose data, not population averages.
          </p>
        </div>
      </footer>

    </div>
  );
}
