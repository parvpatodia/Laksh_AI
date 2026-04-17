# Skill: Testing

## When to Activate

Activate when:
- Writing new features (tests should accompany the feature)
- Fixing bugs (regression test before fixing)
- Refactoring (verify tests exist before changing, run after)
- Developer asks "add tests" or "improve test coverage"
- A test fails and needs diagnosis
- Setting up a test infrastructure for the first time

## Core Principles

1. **Test behavior, not implementation.** Tests verify what the code does for
   its callers, not how it does it internally. If you refactor the internals
   and the tests break, the tests were wrong.

2. **One assertion per test, one concept per test.** A test named
   "test_user_creation" that checks email validation, password hashing,
   AND database insertion is three tests wearing a trench coat.

3. **Tests are documentation.** A new developer should be able to read your
   test names and understand what the system does. Name tests as sentences:
   `it("returns 401 when token is expired")` not `it("test auth")`.

4. **The testing pyramid is not optional.** Many fast unit tests at the base,
   fewer integration tests in the middle, minimal E2E tests at the top.
   Inverting this pyramid (all E2E, no unit tests) creates a slow, brittle,
   expensive test suite.

5. **Deterministic always.** No test should depend on current time, network
   state, random values, or execution order. If you need these, mock them.
   A flaky test is worse than no test — it trains people to ignore failures.

## Patterns and Techniques

### Arrange-Act-Assert (AAA)

Every test follows this structure:
```
// Arrange — set up the preconditions
const user = createTestUser({ role: "admin" });
const request = buildRequest({ userId: user.id });

// Act — perform the action being tested
const result = await handleRequest(request);

// Assert — verify the outcome
expect(result.status).toBe(200);
expect(result.body.user.role).toBe("admin");
```

No test should mix these phases. If you find yourself asserting in the
middle of acting, split into two tests.

### The Bug-First Test (for debugging)

Before fixing a bug:
1. Write a test that reproduces the bug (it should FAIL).
2. Verify the test fails for the right reason.
3. Fix the bug.
4. Verify the test now passes.

This guarantees the bug is actually fixed and cannot regress silently.

### Test Doubles (Mocks, Stubs, Fakes)

Use the lightest double possible:
- **Stub**: Returns canned data. Use when you need a dependency to return
  a specific value. `jest.fn().mockReturnValue(42)`
- **Fake**: A simplified working implementation. Use for complex dependencies
  like databases (in-memory SQLite instead of Postgres) or file systems.
- **Mock**: Verifies interactions. Use sparingly — only when the *call itself*
  is the behavior you're testing (e.g., "verify the email service was called
  with the right recipient"). Over-mocking makes tests brittle.

### What to Test at Each Level

**Unit tests**: Pure functions, business logic, data transformations,
validation rules, state machines. No I/O, no network, no database.

**Integration tests**: API endpoints, database queries, service-to-service
communication, authentication flows. Use real dependencies where feasible
(test database, not mocked).

**E2E tests**: Critical user journeys only. "User can sign up, create a
project, and invite a collaborator." Not every edge case.

### Edge Cases to Always Cover

- Empty inputs (null, undefined, empty string, empty array)
- Boundary values (0, 1, MAX_INT, exactly-at-limit)
- Error paths (network failure, invalid data, unauthorized access)
- Concurrent access (if applicable — two users editing the same resource)

## Checklist

Before considering test work complete:

- [ ] Every new function/endpoint has at least one happy-path test
- [ ] Every bug fix has a regression test that would have caught it
- [ ] Edge cases for the specific domain are covered
- [ ] Tests run in isolation (no shared state between tests)
- [ ] Tests are fast (unit tests < 1s each, integration < 5s each)
- [ ] Test names describe the behavior, not the implementation
- [ ] No commented-out tests or skipped tests without a documented reason
- [ ] Mocks are cleaned up between tests (no leaking state)

## Anti-Patterns

- **Testing private methods directly**: If you need to test a private method,
  it either should be public (extract to a utility) or should be tested
  through the public interface that calls it.
- **Snapshot tests for logic**: Snapshots are for UI rendering, not for
  verifying business logic. A snapshot that says `expect(result).toMatchSnapshot()`
  tells you nothing about what the correct result should be.
- **Test setup longer than the test itself**: If your arrange phase is 30 lines
  and your assert is 1 line, you need test fixtures or factory functions.
- **Testing framework internals**: Don't test that React renders a component,
  that Express routes a request, or that Prisma runs a query. Test YOUR code.
- **100% coverage as a goal**: Coverage measures lines executed, not correctness.
  90% coverage with thoughtful tests beats 100% with trivial assertions.

## Learning Notes

**For the developer**: Testing is a design tool, not just a verification tool.
If code is hard to test, it's usually a sign the code has too many
responsibilities or too-tight coupling. When you find yourself needing 10
mocks to test one function, that's the code telling you it needs refactoring —
not a sign that testing is hard.
