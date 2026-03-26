from fastapi.testclient import TestClient

from api.routes import get_agent
from main import app


class StubAgent:
    def answer(self, question: str) -> dict:
        return {
            "answer": f"Stubbed answer for: {question}",
            "sources": [{"file": "stub.md", "chunk_id": 0, "score": 0.99}],
        }


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_response() -> None:
    app.dependency_overrides[get_agent] = lambda: StubAgent()
    client = TestClient(app)

    response = client.post("/query", json={"question": "How does FastAPI handle async requests?"})
    body = response.json()

    assert response.status_code == 200
    assert "Stubbed answer" in body["answer"]
    assert body["sources"][0]["file"] == "stub.md"

    app.dependency_overrides.clear()
