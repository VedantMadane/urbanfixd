# FastAPI Async Handlers

FastAPI runs on ASGI servers such as Uvicorn. When an endpoint is declared with `async def`,
the handler can await non-blocking operations and return control to the event loop while I/O is pending.

This design improves concurrency for network-heavy workloads. It does not automatically speed up CPU-bound tasks.
For CPU-heavy work, use a task queue or worker process.
