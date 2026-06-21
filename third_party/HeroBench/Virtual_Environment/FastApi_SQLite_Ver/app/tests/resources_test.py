import json

import requests
from deepdiff import DeepDiff

BASE_URL = "http://127.0.0.1:8000/resources"


def test_get_resources():
    with requests.Session() as req_session:
        with open("../Data/resources.json") as j_file:
            items = json.load(j_file)
        response = req_session.get(BASE_URL)
        # Successful get request
        assert response.status_code == 200
        # Same length as json
        assert len(response.json()) == len(items)
        # Check that the response.json() and items contain the same data
        response_items = response.json()
        for item in items:
            assert any(DeepDiff(item, resp_item, ignore_order=True) == {} for resp_item in response_items)


def test_get_items_resources_comparison():
    with requests.Session() as req_session:
        with open("../Data/resources.json") as j_file:
            items = json.load(j_file)
        response = req_session.get(BASE_URL)
        assert response.status_code == 200
        assert len(response.json()) == len(items)
        # Sort both lists by a unique key, for example, "code"
        items_sorted = sorted(items, key=lambda x: x['code'])
        response_items_sorted = sorted(response.json(), key=lambda x: x['code'])
        # Compare each item
        for item, response_item in zip(items_sorted, response_items_sorted):
            assert item == response_item


def test_get_resources_with_filters():
    with requests.Session() as req_session:
        # Test with drop filter
        response = req_session.get(BASE_URL, params={"drop": "copper_ore"})
        assert response.status_code == 200
        assert len(response.json()) >= 1

        # Test with max_level filter
        response = req_session.get(BASE_URL, params={"max_level": 1})
        assert response.status_code == 200
        assert len(response.json()) >= 1

        # Test with min_level filter
        response = req_session.get(BASE_URL, params={"min_level": 1})
        assert response.status_code == 200
        assert len(response.json()) >= 1

        # Test with skill filter
        response = req_session.get(BASE_URL, params={"skill": "mining"})
        assert response.status_code == 200
        assert len(response.json()) >= 1


def test_get_resource():
    with requests.Session() as req_session:
        # Read the test data
        with open("../Data/resources.json") as j_file:
            items = json.load(j_file)

        for item in items:
            code = item["code"]
            response = req_session.get(f"{BASE_URL}/{code}")
            assert response.status_code == 200
            assert DeepDiff(item, response.json(), ignore_order=True) == {}

        # Test for resource not found
        response = req_session.get(f"{BASE_URL}/nonexistentcode")
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Resource not found."


def test_get_resources_not_found():
    with requests.Session() as req_session:
        response = req_session.get(BASE_URL, params={"drop": "nonexistentdrop"})
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Resources not found."
