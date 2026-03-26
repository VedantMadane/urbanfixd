# FastAPI Middleware

Middleware wraps requests and responses globally. It can measure latency, add headers,
perform correlation ID injection, and collect metrics.

Middleware should be lightweight because it executes for every request.
