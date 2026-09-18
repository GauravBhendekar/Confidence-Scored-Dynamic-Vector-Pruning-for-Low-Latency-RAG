import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "active_document" in data
    assert "indexed_chunks" in data

def test_query_endpoint(client):
    response = client.post("/api/query", json={"query": "What is the project title?"})
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "qatm" in data
    assert "token_savings_percent" in data["qatm"]
    assert "query_tier" in data["qatm"]

def test_benchmark_endpoint(client):
    response = client.post("/api/benchmark", json={"dataset_path": "data/evaluation/eval_dataset.json"})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "total_queries_evaluated" in data["summary"]
