"""Session persistence + leaderboard for Laksh.ai.

Dependency-inverted: callers depend on the :class:`SessionStore` ABC, not on
any concrete backend. A local SQLite store is the always-available fallback so
the live demo never breaks; an InsForge-backed store is selected via the
``LAKSH_PERSISTENCE_BACKEND`` flag when the InsForge MCP/key are provisioned.
"""
