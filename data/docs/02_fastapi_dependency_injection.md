# FastAPI Dependency Injection

FastAPI supports dependency injection using `Depends`. A dependency can be a function or class
that provides resources such as database sessions, authentication context, or configuration.

Dependencies can be reused across routes, reducing duplication and centralizing validation logic.
