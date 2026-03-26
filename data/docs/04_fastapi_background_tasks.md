# FastAPI Background Tasks

`BackgroundTasks` lets a route schedule lightweight follow-up work after returning a response.
Typical uses are sending email notifications, writing audit logs, or cleanup steps.

Background tasks run in-process. For durable or high-volume jobs, use an external queue system.
