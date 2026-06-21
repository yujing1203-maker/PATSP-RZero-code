import json

import requests
from deepdiff import DeepDiff

BASE_URL = "http://127.0.0.1:8000/maps"


def test_get_maps():
    with requests.Session() as req_session:
        with open("../Data/maps.json") as j_file:
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


def test_get_maps_individual_comparison():
    with requests.Session() as req_session:
        with open("../Data/maps.json") as j_file:
            items = json.load(j_file)
        response = req_session.get(BASE_URL)
        assert response.status_code == 200
        assert len(response.json()) == len(items)
        # Sort both lists by a unique key, for example, "name"
        items_sorted = sorted(items, key=lambda x: x['name'])
        response_items_sorted = sorted(response.json(), key=lambda x: x['name'])
        # Compare each item
        for item, response_item in zip(items_sorted, response_items_sorted):
            assert item == response_item


def test_get_maps_with_filters():
    with requests.Session() as req_session:
        # Test with content_code filter
        response = req_session.get(BASE_URL, params={"content_code": "ogre"})
        assert response.status_code == 200
        assert len(response.json()) >= 1

        # Test with content_type filter
        response = req_session.get(BASE_URL, params={"content_type": "resource"})
        assert response.status_code == 200
        assert len(response.json()) >= 1


def test_get_map():
    with requests.Session() as req_session:
        # Read the test data
        with open("../Data/maps.json") as j_file:
            items = json.load(j_file)

        for item in items:
            x = item["x"]
            y = item["y"]
            response = req_session.get(f"{BASE_URL}/{x}/{y}")
            assert response.status_code == 200
            assert DeepDiff(item, response.json(), ignore_order=True) == {}

        # Test for map not found
        response = req_session.get(f"{BASE_URL}/999/999")
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Map not found."


def test_get_maps_not_found():
    with requests.Session() as req_session:
        response = req_session.get(BASE_URL, params={"content_code": "nonexistentcode"})
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "Maps not found."
