# /security-audit

Read the security skill from .claude/skills/security.md.

Perform a security audit on this project following the skill's checklist:

1. Scan all files for hardcoded secrets (API keys, passwords, tokens, connection strings).
2. Check all database queries for parameterization (flag any string concatenation in SQL).
3. Review authentication and authorization logic.
4. Check for XSS vectors (dangerouslySetInnerHTML, v-html, direct DOM manipulation with user input).
5. Review CORS configuration.
6. Check dependencies for known vulnerabilities.
7. Review error handling — are stack traces or internal details exposed to clients?
8. Check environment variable usage — are any missing `.env.example` entries?

Write findings to SECURITY_AUDIT.md with severity levels (critical/high/medium/low).

For each finding, include:
- File and line number
- What the issue is
- How to fix it
- Severity and why

Do NOT fix anything automatically. This is a read-only audit.
The developer decides what to address and in what order.
