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
      { label: "Release speed",        unit: "m/s",  sample: "6.8", tier: "good"  },
      { label: "Shot arc",             unit: "deg",  sample: "48",  tier: "elite" },
      { label: "Knee bend at release", unit: "deg",  sample: "138", tier: "good"  },
      { label: "Elbow angle",          unit: "deg",  sample: "162", tier: "elite" },
      { label: "Body sync timing",     unit: "ms",   sample: "290", tier: "good"  },
      { label: "Fluidity score",       unit: "/100", sample: "71",  tier: "good"  },
    ],
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
      { label: "Rep duration",           unit: "s",   sample: "2.4", tier: "good"  },
      { label: "Lowering phase",         unit: "s",   sample: "1.6", tier: "elite" },
      { label: "Lifting phase",          unit: "s",   sample: "0.8", tier: "good"  },
      { label: "Tempo ratio",            unit: "x",   sample: "2.0", tier: "elite" },
      { label: "Range of motion",        unit: "deg", sample: "118", tier: "good"  },
      { label: "Joint tracking quality", unit: "%",   sample: "91",  tier: "elite" },
    ],
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
  { value: "30 fps", label: "Live tracking rate"  },
  { value: "80s",    label: "Analysis time"       },
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
// Joint definitions for the animated skeleton hero
// Basketball jump-shot release pose (simplified 15-point skeleton)
// ---------------------------------------------------------------------------

const SKELETON_JOINTS = [
  // head
  { id: "head",      x: 152, y: 48,  r: 5,   color: "#fbbf24", delay: 0    },
  // torso
  { id: "lShoulder", x: 122, y: 108, r: 5.5, color: "#0ea5e9", delay: 0.1  },
  { id: "rShoulder", x: 188, y: 102, r: 5.5, color: "#0ea5e9", delay: 0.2  },
  { id: "lHip",      x: 130, y: 230, r: 5.5, color: "#10b981", delay: 0.3  },
  { id: "rHip",      x: 175, y: 228, r: 5.5, color: "#10b981", delay: 0.4  },
  // left arm (guide arm)
  { id: "lElbow",    x: 108, y: 162, r: 5,   color: "#0ea5e9", delay: 0.15 },
  { id: "lWrist",    x: 100, y: 205, r: 4,   color: "#38bdf8", delay: 0.25 },
  // right arm (shooting arm -- raised)
  { id: "rElbow",    x: 218, y: 70,  r: 5,   color: "#0ea5e9", delay: 0.35 },
  { id: "rWrist",    x: 228, y: 40,  r: 4,   color: "#38bdf8", delay: 0.05 },
  // left leg
  { id: "lKnee",     x: 118, y: 298, r: 5,   color: "#10b981", delay: 0.45 },
  { id: "lAnkle",    x: 112, y: 368, r: 4,   color: "#34d399", delay: 0.55 },
  // right leg
  { id: "rKnee",     x: 182, y: 294, r: 5,   color: "#10b981", delay: 0.5  },
  { id: "rAnkle",    x: 185, y: 366, r: 4,   color: "#34d399", delay: 0.6  },
] as const;

const SKELETON_CONNECTIONS: [string, string, string][] = [
  // spine / torso
  ["head",      "lShoulder", "#0ea5e9"],
  ["head",      "rShoulder", "#0ea5e9"],
  ["lShoulder", "rShoulder", "#0ea5e9"],
  ["lShoulder", "lHip",      "#10b981"],
  ["rShoulder", "rHip",      "#10b981"],
  ["lHip",      "rHip",      "#10b981"],
  // left arm
  ["lShoulder", "lElbow",    "#0ea5e9"],
  ["lElbow",    "lWrist",    "#38bdf8"],
  // shooting arm
  ["rShoulder", "rElbow",    "#0ea5e9"],
  ["rElbow",    "rWrist",    "#38bdf8"],
  // legs
  ["lHip",      "lKnee",     "#10b981"],
  ["lKnee",     "lAnkle",    "#34d399"],
  ["rHip",      "rKnee",     "#10b981"],
  ["rKnee",     "rAnkle",    "#34d399"],
];

// ---------------------------------------------------------------------------
// Animated skeleton hero visual (server component -- pure SVG + CSS)
// ---------------------------------------------------------------------------

function AnimatedSkeletonHero() {
  const jointMap = Object.fromEntries(SKELETON_JOINTS.map((j) => [j.id, j]));

  return (
    <div className="relative hidden lg:flex items-center justify-center select-none">
      {/* Outer glow rings */}
      <div className="absolute w-72 h-72 rounded-full border border-brand-500/10 animate-ping"
           style={{ animationDuration: "3s" }} />
      <div className="absolute w-96 h-96 rounded-full border border-brand-500/5 animate-ping"
           style={{ animationDuration: "4s", animationDelay: "1s" }} />

      {/* HUD frame */}
      <div className="relative w-80 h-[480px]">
        {/* Corner brackets */}
        <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-brand-500/60 rounded-tl" />
        <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-brand-500/60 rounded-tr" />
        <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-brand-500/60 rounded-bl" />
        <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-brand-500/60 rounded-br" />

        {/* Scan line animation */}
        <div
          className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-brand-500/50 to-transparent pointer-events-none"
          style={{ animation: "scanLine 3s linear infinite" }}
        />

        {/* HUD label */}
        <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
          <span className="text-[9px] font-mono font-bold tracking-widest text-brand-500/70 uppercase">
            Pose tracking
          </span>
          <span className="flex items-center gap-1.5 text-[9px] font-mono text-emerald-400/80">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Live
          </span>
        </div>

        {/* Main skeleton SVG */}
        <svg
          viewBox="0 0 310 440"
          className="w-full h-full"
          style={{ animation: "floatY 6s ease-in-out infinite" }}
        >
          {/* Basketball arc trail */}
          <path
            d="M228 40 Q280 -20 310 20"
            fill="none"
            stroke="#f59e0b"
            strokeWidth={1}
            strokeDasharray="4 4"
            opacity={0.4}
          />

          {/* Basketball */}
          <g style={{ animation: "ballFloat 6s ease-in-out infinite" }}>
            <circle cx={244} cy={22} r={16} fill="none" stroke="#f59e0b" strokeWidth={2} opacity={0.9} />
            <circle cx={244} cy={22} r={16} fill="#f59e0b" opacity={0.08} />
            {/* Ball lines */}
            <line x1={228} y1={22} x2={260} y2={22} stroke="#f59e0b" strokeWidth={1} opacity={0.5} />
            <path d="M244 6 Q257 22 244 38" fill="none" stroke="#f59e0b" strokeWidth={1} opacity={0.5} />
            <path d="M244 6 Q231 22 244 38" fill="none" stroke="#f59e0b" strokeWidth={1} opacity={0.5} />
            {/* Glow */}
            <circle cx={244} cy={22} r={20} fill="none" stroke="#f59e0b" strokeWidth={4} opacity={0.1} />
          </g>

          {/* Skeleton connections */}
          {SKELETON_CONNECTIONS.map(([a, b, stroke]) => {
            const j1 = jointMap[a];
            const j2 = jointMap[b];
            if (!j1 || !j2) return null;
            return (
              <line
                key={`${a}-${b}`}
                x1={j1.x} y1={j1.y}
                x2={j2.x} y2={j2.y}
                stroke={stroke}
                strokeWidth={1.8}
                strokeLinecap="round"
                opacity={0.45}
              />
            );
          })}

          {/* Joint dots */}
          {SKELETON_JOINTS.map((j) => (
            <g key={j.id}>
              {/* Outer pulse ring */}
              <circle
                cx={j.x} cy={j.y} r={j.r + 5}
                fill="none"
                stroke={j.color}
                strokeWidth={0.8}
                opacity={0.3}
                style={{ animation: `pulseRing 2s ease-in-out ${j.delay}s infinite` }}
              />
              {/* Mid ring */}
              <circle
                cx={j.x} cy={j.y} r={j.r + 2}
                fill="none"
                stroke={j.color}
                strokeWidth={0.5}
                opacity={0.5}
              />
              {/* Core dot */}
              <circle cx={j.x} cy={j.y} r={j.r} fill={j.color} opacity={0.95} />
              {/* Inner highlight */}
              <circle cx={j.x - 1} cy={j.y - 1} r={j.r * 0.4} fill="white" opacity={0.3} />
            </g>
          ))}

          {/* Measurement annotation -- elbow angle */}
          <g opacity={0.7}>
            <path
              d="M218 70 Q230 80 228 40"
              fill="none" stroke="#0ea5e9" strokeWidth={1} strokeDasharray="2 2"
            />
            <rect x={232} y={52} width={42} height={14} rx={3}
                  fill="#0c1524" stroke="#0ea5e9" strokeWidth={0.5} opacity={0.9} />
            <text x={253} y={63} textAnchor="middle"
                  fill="#38bdf8" fontSize={9} fontFamily="monospace" fontWeight="bold">
              162 deg
            </text>
          </g>

          {/* Measurement annotation -- knee bend */}
          <g opacity={0.7}>
            <line x1={182} y1={294} x2={210} y2={310}
                  stroke="#10b981" strokeWidth={1} strokeDasharray="2 2" />
            <rect x={212} y={302} width={48} height={14} rx={3}
                  fill="#0c1524" stroke="#10b981" strokeWidth={0.5} opacity={0.9} />
            <text x={236} y={313} textAnchor="middle"
                  fill="#34d399" fontSize={9} fontFamily="monospace" fontWeight="bold">
              138 deg
            </text>
          </g>

          {/* Wrist velocity label */}
          <g opacity={0.7}>
            <line x1={228} y1={40} x2={260} y2={28}
                  stroke="#f59e0b" strokeWidth={1} strokeDasharray="2 2" />
            <rect x={261} y={18} width={50} height={14} rx={3}
                  fill="#0c1524" stroke="#f59e0b" strokeWidth={0.5} opacity={0.9} />
            <text x={286} y={29} textAnchor="middle"
                  fill="#fbbf24" fontSize={9} fontFamily="monospace" fontWeight="bold">
              6.8 m/s
            </text>
          </g>
        </svg>

        {/* Bottom HUD bar */}
        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex items-center justify-between text-[8px] font-mono text-slate-600">
            <span>33 landmarks tracked</span>
            <span className="text-brand-500/60">84% confidence</span>
          </div>
          <div className="mt-1 h-0.5 rounded-full bg-surface-700 overflow-hidden">
            <div className="h-full bg-brand-500/60 rounded-full" style={{ width: "84%" }} />
          </div>
        </div>
      </div>

      {/* Keyframe definitions -- injected once into the SVG context */}
      <style>{`
        @keyframes scanLine {
          0%   { top: 0%; opacity: 0; }
          5%   { opacity: 1; }
          95%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes floatY {
          0%, 100% { transform: translateY(0px); }
          50%       { transform: translateY(-8px); }
        }
        @keyframes ballFloat {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50%       { transform: translateY(-5px) rotate(8deg); }
        }
        @keyframes pulseRing {
          0%, 100% { opacity: 0.15; transform: scale(1); }
          50%       { opacity: 0.5;  transform: scale(1.3); }
        }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Basketball court background SVG (decorative, used inside sport card)
// ---------------------------------------------------------------------------

function BasketballCardBg() {
  return (
    <svg
      viewBox="0 0 400 260"
      className="absolute inset-0 w-full h-full opacity-[0.04] pointer-events-none"
      preserveAspectRatio="xMidYMid slice"
    >
      {/* Half-court line */}
      <line x1={200} y1={0} x2={200} y2={260} stroke="white" strokeWidth={2} />
      {/* Centre circle */}
      <circle cx={200} cy={130} r={50} fill="none" stroke="white" strokeWidth={2} />
      {/* Three-point arc (right) */}
      <path d="M320 10 Q400 130 320 250" fill="none" stroke="white" strokeWidth={2} />
      {/* Key (right) */}
      <rect x={300} y={80} width={100} height={100} fill="none" stroke="white" strokeWidth={2} />
      {/* Free throw circle */}
      <circle cx={300} cy={130} r={30} fill="none" stroke="white" strokeWidth={1.5} />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Gym / barbell background SVG (decorative)
// ---------------------------------------------------------------------------

function GymCardBg() {
  return (
    <svg
      viewBox="0 0 400 260"
      className="absolute inset-0 w-full h-full opacity-[0.04] pointer-events-none"
      preserveAspectRatio="xMidYMid slice"
    >
      {/* Barbell */}
      <rect x={80} y={120} width={240} height={14} rx={7} fill="white" />
      {/* Left plates */}
      <rect x={50}  y={100} width={32} height={54} rx={4} fill="white" />
      <rect x={20}  y={108} width={30} height={38} rx={4} fill="white" />
      {/* Right plates */}
      <rect x={318} y={100} width={32} height={54} rx={4} fill="white" />
      <rect x={350} y={108} width={30} height={38} rx={4} fill="white" />
      {/* ROM arc */}
      <path d="M140 180 Q200 60 260 180" fill="none" stroke="white" strokeWidth={2} strokeDasharray="6 4" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

type Tier = "elite" | "good" | "none";

function TierBadge({ tier }: { tier: Tier }) {
  if (tier === "elite") {
    return (
      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded
                       bg-emerald-900/50 text-emerald-400 border border-emerald-700/50">
        ELITE
      </span>
    );
  }
  if (tier === "good") {
    return (
      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded
                       bg-sky-900/50 text-sky-400 border border-sky-700/50">
        GOOD
      </span>
    );
  }
  return null;
}

function MetricRow({
  label, unit, sample, tier,
}: { label: string; unit: string; sample: string; tier: Tier }) {
  return (
    <div className="flex items-center justify-between py-2
                    border-b border-surface-700/40 last:border-0 gap-3">
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
        {/* Background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-surface-950 via-surface-900 to-surface-850" />
        {/* Radial accent glow */}
        <div className="absolute -top-60 -left-60 w-[700px] h-[700px] rounded-full
                        bg-brand-500/6 blur-3xl pointer-events-none" />
        <div className="absolute top-20 right-0 w-[500px] h-[500px] rounded-full
                        bg-emerald-500/3 blur-3xl pointer-events-none" />
        {/* Subtle grid texture */}
        <div
          className="absolute inset-0 opacity-[0.025] pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,.2) 1px, transparent 1px), " +
              "linear-gradient(90deg, rgba(255,255,255,.2) 1px, transparent 1px)",
            backgroundSize: "64px 64px",
          }}
        />

        <div className="relative z-10 max-w-screen-2xl mx-auto px-6 xl:px-16
                        pt-20 pb-12 lg:pt-28 lg:pb-16
                        grid grid-cols-1 lg:grid-cols-2 gap-12 xl:gap-20 items-center">

          {/* Left: headline */}
          <div>
            {/* Eyebrow */}
            <div className="inline-flex items-center gap-2 mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />
              <span className="text-[11px] font-bold uppercase tracking-[0.25em] text-brand-400">
                Sports Biomechanics Research
              </span>
            </div>

            <h1 className="headline-display text-5xl sm:text-6xl lg:text-6xl xl:text-7xl
                           font-black text-white mb-8">
              Read your<br />
              body<br />
              <span className="text-brand-500">in motion.</span>
            </h1>

            <p className="text-lg xl:text-xl text-slate-400 leading-relaxed max-w-xl mb-12">
              33-point skeletal tracking. Real measurements derived from your camera.
              Personalised coaching generated from what your body actually does -- not
              from population averages.
            </p>

            <div className="flex flex-wrap gap-4">
              <Link
                href="/basketball"
                className="inline-flex items-center gap-2.5 rounded-xl bg-brand-500 hover:bg-brand-400
                           text-white font-semibold text-sm px-7 py-3.5 transition-all duration-200
                           shadow-lg shadow-brand-500/30 hover:shadow-brand-500/50 hover:-translate-y-0.5"
              >
                Analyse a jump shot
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24"
                     stroke="currentColor" strokeWidth={2.5}>
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

          {/* Right: animated skeleton hero */}
          <AnimatedSkeletonHero />
        </div>

        {/* Stats bar */}
        <div className="relative z-10 border-t border-surface-700/50
                        bg-surface-850/70 backdrop-blur-sm">
          <div className="max-w-screen-2xl mx-auto px-6 xl:px-16 py-5">
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-surface-700/50">
              {METHODOLOGY_STATS.map(({ value, label }) => (
                <div key={label} className="text-center px-4 sm:px-8 first:pl-0 last:pr-0
                                            flex flex-col items-center gap-1">
                  <p className="stat-number text-2xl xl:text-3xl text-white">{value}</p>
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                    {label}
                  </p>
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
            {SPORTS.map((sport, si) => (
              <Link
                key={sport.id}
                href={sport.href}
                className={`sport-card group block rounded-2xl border border-surface-700
                            bg-surface-800 overflow-hidden
                            ${si === 0 ? "hover:border-brand-500/50" : "hover:border-perf-500/40"}`}
              >
                {/* Decorative background illustration */}
                {si === 0 ? <BasketballCardBg /> : <GymCardBg />}

                {/* Top gradient accent line */}
                <div className={`h-0.5 w-full bg-gradient-to-r
                  ${si === 0 ? "from-brand-500/60 via-brand-500/20 to-transparent"
                             : "from-perf-500/60 via-perf-500/20 to-transparent"}`}
                />

                <div className="relative p-7 xl:p-8">
                  {/* Sport header */}
                  <div className="flex items-start justify-between mb-5">
                    <div className="flex items-center gap-4">
                      {/* Icon badge */}
                      <div className={`rounded-xl border p-3 shrink-0
                        ${si === 0
                          ? "bg-brand-500/10 border-brand-500/30 text-brand-500"
                          : "bg-perf-500/10 border-perf-500/30 text-perf-500"}`}
                      >
                        {si === 0 ? (
                          <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7"
                               stroke="currentColor" strokeWidth={1.5}>
                            <circle cx="12" cy="12" r="9" />
                            <path strokeLinecap="round" d="M3.6 9h16.8M3.6 15h16.8" />
                            <path strokeLinecap="round" d="M12 3a15 15 0 0 1 4 9 15 15 0 0 1-4 9M12 3a15 15 0 0 0-4 9 15 15 0 0 0 4 9" />
                          </svg>
                        ) : (
                          <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7"
                               stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round"
                                  d="M6 8h2m8 0h2M8 8V6a1 1 0 0 1 1-1h1m4 0h1a1 1 0 0 1 1 1v2M8 8h8m0 0v8M8 8v8m0 0H6a1 1 0 0 1-1-1v-1m11 2h2a1 1 0 0 0 1-1v-1" />
                            <rect x="9" y="8" width="6" height="8" rx="0.5" />
                          </svg>
                        )}
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
                    <div className={`rounded-full p-2 shrink-0 mt-1 transition-all duration-200
                      bg-surface-700/50 text-slate-600
                      group-hover:text-brand-500 group-hover:bg-brand-500/10`}>
                      <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform"
                           fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </div>

                  <p className="text-sm text-slate-400 leading-relaxed mb-6">{sport.description}</p>

                  {/* Metric preview table */}
                  <div className="rounded-xl bg-surface-900/80 border border-surface-700/40 px-5 py-1">
                    <p className="label-section pt-3 pb-1">Measured outputs</p>
                    {sport.metrics.map((m) => (
                      <MetricRow
                        key={m.label}
                        label={m.label} unit={m.unit} sample={m.sample} tier={m.tier as Tier}
                      />
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
                {/* Step icon */}
                <div className="mb-5">
                  {i === 0 && (
                    <div className="w-10 h-10 rounded-xl bg-brand-500/10 border border-brand-500/30
                                    flex items-center justify-center mb-4">
                      <svg className="w-5 h-5 text-brand-500" fill="none" viewBox="0 0 24 24"
                           stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round"
                              d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                  )}
                  {i === 1 && (
                    <div className="w-10 h-10 rounded-xl bg-perf-500/10 border border-perf-500/30
                                    flex items-center justify-center mb-4">
                      <svg className="w-5 h-5 text-perf-500" fill="none" viewBox="0 0 24 24"
                           stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round"
                              d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
                      </svg>
                    </div>
                  )}
                  {i === 2 && (
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30
                                    flex items-center justify-center mb-4">
                      <svg className="w-5 h-5 text-emerald-500" fill="none" viewBox="0 0 24 24"
                           stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round"
                              d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                  )}
                  <div className="text-5xl xl:text-6xl font-black font-mono text-surface-700
                                  select-none leading-none">
                    {step.step}
                  </div>
                </div>
                <h3 className="text-base font-bold text-white mb-3">{step.title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================== */}
      {/* LIVE PREVIEW STRIP -- mini skeleton thumbnails                      */}
      {/* ================================================================== */}
      <section className="py-12 border-t border-surface-700/40 bg-surface-850/40">
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16">
          <div className="flex flex-col lg:flex-row items-center gap-10 xl:gap-16">

            {/* Text */}
            <div className="lg:w-1/2 xl:w-2/5">
              <span className="label-section block mb-4">Real-time pose overlay</span>
              <h3 className="text-2xl xl:text-3xl font-black text-white mb-4">
                See your skeleton before you record.
              </h3>
              <p className="text-sm text-slate-500 leading-relaxed mb-6">
                The live camera view draws your 33-point skeleton in real time at 30 fps.
                Joint confidence is shown as a percentage badge -- if tracking quality drops,
                you see it immediately and can reposition before you record.
              </p>
              <Link
                href="/basketball"
                className="inline-flex items-center gap-2 text-sm font-semibold text-brand-400
                           hover:text-brand-300 transition-colors"
              >
                Try it now
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24"
                     stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            </div>

            {/* Mini skeleton strip */}
            <div className="lg:w-1/2 xl:w-3/5 grid grid-cols-3 gap-3">
              {[
                { label: "Pose 92%", color: "#10b981", dots: ["#10b981","#0ea5e9","#10b981","#38bdf8","#10b981"] },
                { label: "Pose 78%", color: "#0ea5e9", dots: ["#0ea5e9","#38bdf8","#0ea5e9","#10b981","#38bdf8"] },
                { label: "Pose 0%",  color: "#f43f5e", dots: ["#334155","#334155","#334155","#334155","#334155"] },
              ].map(({ label, color, dots }) => (
                <div
                  key={label}
                  className="rounded-xl border border-surface-700/60 bg-surface-800
                             aspect-[3/4] flex flex-col items-center justify-center gap-2 relative overflow-hidden"
                >
                  {/* Mock skeleton dots */}
                  <div className="flex flex-col items-center gap-1.5">
                    {/* Head */}
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: dots[0] }} />
                    {/* Shoulders */}
                    <div className="flex gap-3">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[1] }} />
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[1] }} />
                    </div>
                    {/* Elbows */}
                    <div className="flex gap-5">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[2] }} />
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[2] }} />
                    </div>
                    {/* Hips */}
                    <div className="flex gap-2.5">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[3] }} />
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[3] }} />
                    </div>
                    {/* Knees */}
                    <div className="flex gap-3.5">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[4] }} />
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dots[4] }} />
                    </div>
                  </div>
                  {/* Badge */}
                  <div
                    className="absolute top-2 left-2 text-[9px] font-mono font-bold px-1.5 py-0.5 rounded"
                    style={{ color, backgroundColor: `${color}20`, border: `1px solid ${color}50` }}
                  >
                    {label}
                  </div>
                  {/* No-pose warning on last card */}
                  {label === "Pose 0%" && (
                    <div className="absolute bottom-2 left-2 right-2 text-center text-[8px]
                                    text-rose-400 font-medium bg-rose-950/80 rounded py-0.5 px-1">
                      Step into frame
                    </div>
                  )}
                </div>
              ))}
            </div>
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
                  <span className="shrink-0 text-[9px] font-bold uppercase tracking-wider
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
        <div className="max-w-screen-2xl mx-auto px-6 xl:px-16
                        flex flex-col sm:flex-row items-center justify-between gap-4">
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
