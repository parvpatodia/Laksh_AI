# Skill: Performance Optimization

## When to Activate

Activate when:
- User reports slowness or latency issues
- Working on database queries touching large datasets
- Optimizing API response times
- Improving frontend load times or interaction responsiveness
- Profiling or benchmarking existing code
- Designing systems that must handle high concurrency or throughput
- Background review task flags performance concerns

## Core Principles

1. **Measure before optimizing.** Never optimize based on intuition. Profile
   first, identify the actual bottleneck, then optimize that specific thing.
   The bottleneck is almost never where you think it is.

2. **Optimize the algorithm, not the code.** Switching from O(n²) to O(n log n)
   beats micro-optimizing the inner loop of an O(n²) algorithm every time.
   Think about data structures and algorithmic complexity before touching
   implementation details.

3. **The fastest code is code that doesn't run.** Cache computed results.
   Avoid redundant calculations. Short-circuit early. Lazy-load what isn't
   needed yet. Paginate instead of loading everything.

4. **Latency has a budget.** Know your targets before optimizing. A 200ms API
   response is fine for most apps. Shaving it to 50ms when no user will notice
   is wasted effort. Optimization without a target is just tinkering.

5. **Premature optimization is real.** Don't optimize during initial development
   unless you know from domain experience that a specific path will be hot.
   Write clear code first, measure in realistic conditions, optimize what matters.

## Patterns and Techniques

### Database Performance

- **Indexing**: Add indexes on columns used in WHERE, JOIN, and ORDER BY.
  Compound indexes for multi-column queries (order matters — most selective
  column first).
- **N+1 queries**: The most common database performance bug. If you're running
  a query inside a loop, you have an N+1. Use eager loading, joins, or batch
  fetching.
- **Query analysis**: Use EXPLAIN/EXPLAIN ANALYZE to understand query plans.
  Look for sequential scans on large tables, missing indexes, and
  unnecessary sorts.
- **Connection pooling**: Don't open a new database connection per request.
  Use a pool (PgBouncer, Prisma's built-in pool, SQLAlchemy pool).
- **Pagination**: Never `SELECT *` without a LIMIT. Use cursor-based pagination
  for large datasets (keyset pagination), not OFFSET which degrades linearly.

### API and Backend Performance

- **Caching layers**: In-memory (LRU cache) for hot data, Redis/Memcached for
  shared cache, CDN for static assets, HTTP cache headers for client caching.
- **Async where possible**: Don't block a request handler waiting for email
  sending, PDF generation, or webhook delivery. Push to a job queue.
- **Payload size**: Return only the fields the client needs. GraphQL's
  strength, but achievable in REST with sparse fieldsets or dedicated
  endpoints.
- **Compression**: gzip/brotli for API responses. Usually handled by the
  reverse proxy (nginx, Cloudflare) but verify it's active.

### Frontend Performance

- **Bundle size**: Tree-shake, code-split, lazy-load routes and heavy
  components. Measure with `npx bundlephobia` or webpack-bundle-analyzer.
- **Rendering**: Avoid unnecessary re-renders. In React: memoize expensive
  components, use `useMemo`/`useCallback` where profiling shows it helps
  (not everywhere), virtualize long lists.
- **Core Web Vitals**: LCP (largest contentful paint) < 2.5s, FID/INP < 200ms,
  CLS < 0.1. Measure with Lighthouse or web-vitals library.
- **Images**: Use modern formats (WebP, AVIF), serve responsive sizes,
  lazy-load below-the-fold images.

### Profiling Tools

- **Node.js**: `--prof` flag, clinic.js, 0x (flame graphs)
- **Python**: cProfile, py-spy, line_profiler
- **Frontend**: Chrome DevTools Performance tab, Lighthouse, React DevTools Profiler
- **Database**: EXPLAIN ANALYZE, pg_stat_statements, slow query log

## Checklist

Before considering performance work complete:

- [ ] Bottleneck was identified through measurement, not assumption
- [ ] Before/after metrics are recorded (exact numbers, not "feels faster")
- [ ] The optimization targets the actual bottleneck, not a secondary concern
- [ ] No functionality was broken (tests still pass)
- [ ] The optimization doesn't make the code significantly harder to maintain
- [ ] Edge cases still perform acceptably (empty data, max data, concurrent load)
- [ ] If caching was added: invalidation strategy is defined and tested

## Anti-Patterns

- **Premature caching**: Adding Redis before you've measured whether the
  database query is actually slow. Often an index is the real fix.
- **Over-memoization**: `useMemo` on every value in a React component adds
  overhead. Only memoize what profiling shows is expensive.
- **Micro-benchmarks as proof**: Benchmarking an isolated function in a loop
  doesn't reflect real-world performance with I/O, GC pressure, and
  contention. Benchmark the actual endpoint or user flow.
- **Optimizing cold paths**: Spending a week optimizing the admin settings page
  that 3 people use monthly while the dashboard (hit 10k times/day) is slow.

## Learning Notes

**For the developer**: The mental model that matters most is understanding where
time is actually spent. In a typical web request: network latency (~50-200ms),
DNS + TLS (~20-50ms), server processing (~10-100ms), database queries (~1-50ms
each). If your database query takes 2ms and your API response takes 800ms, the
database is not your problem — look at the middleware stack, serialization, or
network. Flame graphs are the single most useful tool for seeing where time goes.
Learn to read them.
