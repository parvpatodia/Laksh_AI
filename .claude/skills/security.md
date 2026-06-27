# Skill: Security Audit

## When to Activate

Activate when:
- Implementing authentication, authorization, or session management
- Handling user input (forms, API parameters, file uploads)
- Working with databases (queries, migrations, access control)
- Dealing with secrets, API keys, tokens, or environment variables
- Building payment or financial features
- Developer asks for a security review
- Automated review task detects security-adjacent code changes

## Core Principles

1. **Never trust input.** All input — from users, APIs, databases, files,
   environment variables, URL parameters — is hostile until validated.
   Validate on the server, not just the client.

2. **Least privilege everywhere.** Every component gets the minimum permissions
   it needs to function. Database users get read-only access unless they need
   write. API tokens are scoped to specific operations. File permissions are
   restrictive by default.

3. **Defense in depth.** No single security measure is sufficient. Layer
   them: input validation AND parameterized queries AND output encoding AND
   CSP headers. If one layer fails, the others catch it.

4. **Secrets never touch code.** No API keys, passwords, tokens, or
   certificates in source code, commit history, logs, error messages, or
   client-side bundles. Environment variables or a secrets manager, always.

5. **Fail closed.** When a security check encounters an error, the default
   is to deny access, not to grant it. An auth middleware that crashes
   should return 403, not pass the request through.

## Patterns and Techniques

### Input Validation

Validate using schemas, not manual checks:
- Use Zod, Joi, or similar for runtime validation
- Define the schema as a whitelist (what IS allowed), not a blacklist
  (what IS NOT allowed)
- Validate type, length, format, and range
- Reject early, before any processing

### SQL Injection Prevention

Use parameterized queries exclusively. No string concatenation for SQL:
- ORMs (Prisma, Drizzle, SQLAlchemy) handle this by default
- Raw queries must use parameter binding (`$1`, `?`, `:param`)
- Stored procedures are also parameterized if using user input

### XSS Prevention

- Use framework auto-escaping (React's JSX, Next.js, Vue templates)
- Never use `dangerouslySetInnerHTML`, `v-html`, or `innerHTML` with
  user-provided content
- Implement Content-Security-Policy headers
- Sanitize HTML if you must accept rich text (DOMPurify)

### Authentication Checklist

- Passwords: bcrypt or argon2, never MD5/SHA1/SHA256 alone
- Sessions: HttpOnly, Secure, SameSite cookies
- Tokens: Short-lived access tokens, longer-lived refresh tokens
- Rate limiting on login attempts
- Account lockout after repeated failures
- Constant-time comparison for secrets (prevent timing attacks)

### Authorization Patterns

- Check permissions on every request, not just at the UI level
- Verify resource ownership (user A cannot access user B's data)
- Use role-based (RBAC) or attribute-based (ABAC) access control
- Never rely on client-side route guards as your only protection

### Secrets Management

- Environment variables for configuration
- A secrets manager (Vault, AWS Secrets Manager, Doppler) for production
- .env files for local development, NEVER committed to git
- Rotate secrets on any suspected exposure
- Different secrets for dev/staging/production

## Checklist

Security review checklist for any code change:

- [ ] All user input is validated with a schema
- [ ] Database queries use parameterization (no string concat)
- [ ] No secrets in source code, logs, or error messages
- [ ] Authentication checks are present on protected routes
- [ ] Authorization verifies resource ownership, not just auth status
- [ ] Error messages do not leak internal details (stack traces, DB schema)
- [ ] HTTPS is enforced in production
- [ ] CORS is configured restrictively (not `*`)
- [ ] Rate limiting is in place on auth endpoints and expensive operations
- [ ] File uploads are validated (type, size, content), not just by extension
- [ ] Dependencies have no known critical vulnerabilities (`npm audit`, `pip audit`)
- [ ] Sensitive data is not logged or stored in plaintext

## Anti-Patterns

- **Security through obscurity**: Hiding an admin panel at `/admin-secret-path`
  is not security. It will be found.
- **Client-side only validation**: The server must validate independently.
  Client validation is a UX convenience, not a security measure.
- **Rolling your own crypto**: Use established libraries for encryption,
  hashing, and token generation. Homemade crypto has homemade vulnerabilities.
- **Overly broad error messages to "help" users**: "Invalid password for
  user@example.com" confirms the account exists. Return "Invalid credentials."
- **Disabling security in development**: If CORS, CSP, or auth middleware
  is disabled in dev, you will ship without it eventually.

## Learning Notes

**For the developer**: Security is not a feature you add later — it's a
property of how the system is built. The OWASP Top 10 is worth reading once
a year. Most breaches exploit basic issues (injection, broken auth, exposed
data), not sophisticated zero-days. Getting the basics right blocks the
vast majority of real-world attacks.
