import pytest
import httpx
import json
from unittest.mock import MagicMock, patch
from temp.vikunja_9b import VikunjaClient, make_request, normalize_url, build_headers, VikunjaError

# Mock config
@pytest.fixture
def mock_config():
    return {
        "url": "http://localhost:3456/api/v1",
        "token": "test-token",
        "api_version": "v1",
        "retries": 0
    }

@pytest.fixture
def mock_client():
    return MagicMock(spec=httpx.Client)

def test_normalize_url():
    assert normalize_url("http://host:3456/api/v1") == "http://host:3456/api"
    assert normalize_url("http://host:3456/v1") == "http://host:3456"
    assert normalize_url("http://host:3456/api") == "http://host:3456/api"
    assert normalize_url("http://host:3456/") == "http://host:3456"

def test_build_headers(mock_config):
    headers = build_headers(mock_config)
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["Content-Type"] == "application/json"

def test_make_request_success(mock_client, mock_config):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"id": 1, "title": "test"}'
    mock_response.json.return_value = {"id": 1, "title": "test"}
    mock_client.get.return_value = mock_response

    result = make_request(mock_client, mock_config, "GET", "/projects")
    assert result["title"] == "test"

def test_make_request_error(mock_client, mock_config):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"message": "not found"}'
    mock_response.json.return_value = {"message": "not found"}

    # Simulate HTTPStatusError
    error = httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
    mock_client.get.side_effect = error

    with pytest.raises(VikunjaError) as excinfo:
        make_request(mock_client, mock_config, "GET", "/projects")
    assert "not found" in str(excinfo.value)

def test_vikunja_client_get_projects(mock_client, mock_config):
    with patch("temp.vikunja_9b.make_request") as mock_make:
        mock_make.return_value = [{"id": 1}]
        client = VikunjaClient(mock_config)
        projects = client.get_projects()
        assert len(projects) == 1
        assert projects[0]["id"] == 1
