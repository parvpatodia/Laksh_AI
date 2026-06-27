import Leaderboard from "@/components/Leaderboard";

export const metadata = {
  title: "Leaderboard -- Laksh.ai",
  description:
    "Top form indices per exercise. The form index is a transparent, " +
    "uncalibrated relative ranking measured from real movement -- not a graded score.",
};

export default function LeaderboardPage() {
  return <Leaderboard />;
}
