# FastAPI Validation

Request bodies are validated using Pydantic models. Type hints in route signatures define expected fields,
constraints, and nested structure.

If validation fails, FastAPI returns a 422 response with detailed error locations and messages.
