from fastapi.testclient import TestClient
from src.inference import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_performance_endpoint():
    response = client.get("/performance")

    assert response.status_code == 200