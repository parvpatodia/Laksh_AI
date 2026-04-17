# /suggest-next

Read GOALS.md to understand the project's milestones and current phase.
Read PROJECT_CONTEXT.md for stack and constraints.
Review recent git log (last 10 commits) to see what was done recently.
Check for any URGENT.md, HEALTH_LOG.md, or REVIEW_LOG.md files.

Based on all of this context, answer the developer's question:
"What should I work on next?"

Provide exactly 3 recommendations, ranked by impact:

1. **Highest leverage**: The single task that moves the current milestone
   forward the most. Explain why.
2. **Risk reduction**: A tech debt item, bug, or security issue that will
   become more expensive the longer it's deferred. Explain the cost of delay.
3. **Quick win**: Something achievable in under an hour that improves quality,
   DX, or user experience. Explain the payoff.

For each recommendation, include estimated effort (S/M/L) and which files
or areas of the codebase are involved.

Keep it concise. The developer wants actionable direction, not a report.
