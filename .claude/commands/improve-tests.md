# /improve-tests

Read the testing skill from .claude/skills/testing.md.

Analyze the current test coverage and quality:

1. Run the existing test suite. Record what passes and fails.
2. Identify files and functions with no test coverage.
3. Identify tests that are weak (testing implementation details, no meaningful
   assertions, snapshot-heavy with no logic tests).
4. Prioritize what to test based on risk:
   - Business-critical paths (auth, payments, core features) — highest priority
   - Data transformation and validation logic — high priority
   - API endpoints and integrations — medium priority
   - UI components with logic — medium priority
   - Pure utility functions — lower priority (but fastest to write)

Present your findings, then ask the developer:
"Which area should I write tests for? Here are my top 3 recommendations
ranked by risk coverage."

After the developer chooses, write the tests following the testing skill's
AAA pattern. Run them to verify they pass. If any fail, diagnose whether
the test is wrong or the code has a bug — and report which.
