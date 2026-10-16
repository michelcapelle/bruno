import json
import tempfile
import unittest
from pathlib import Path

from bruno.app import create_app


class BrunoApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.app = create_app(database_path)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_homepage_and_api_index_are_available(self):
        homepage = self.client.get("/")
        api_index = self.client.get("/api")

        self.assertEqual(homepage.status_code, 200)
        self.assertIn(b"Trace the life", homepage.data)
        self.assertEqual(
            api_index.json,
            {"concepts": "/api/concepts", "thinkers": "/api/thinkers"},
        )

    def test_concepts_support_crud_and_validation(self):
        created = self.client.post(
            "/api/concepts",
            data=json.dumps({"name": "  Public sphere ", "description": "A shared space"}),
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json["name"], "Public sphere")
        self.assertEqual(created.json["description"], "A shared space")
        item_id = created.json["id"]

        updated = self.client.patch(
            f"/api/concepts/{item_id}",
            json={"description": "A space for public discourse"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["description"], "A space for public discourse")

        listed = self.client.get("/api/concepts")
        self.assertEqual(len(listed.json), 1)
        self.assertEqual(self.client.get(f"/api/concepts/{item_id}").status_code, 200)

        invalid = self.client.post("/api/concepts", json={"name": " "})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("error", invalid.json)

        deleted = self.client.delete(f"/api/concepts/{item_id}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/api/concepts/{item_id}").status_code, 404)

    def test_thinkers_have_separate_persistent_records(self):
        created = self.client.post(
            "/api/thinkers",
            json={"name": "Hannah Arendt", "biography": "Political theorist"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json["biography"], "Political theorist")
        self.assertEqual(self.client.get("/api/thinkers").json[0]["name"], "Hannah Arendt")
        self.assertEqual(self.client.get("/api/concepts").json, [])

    def test_records_survive_application_recreation(self):
        self.client.post("/api/concepts", json={"name": "Plurality"})

        restarted_app = create_app(Path(self.temp_dir.name) / "test.sqlite3")
        restarted_client = restarted_app.test_client()

        self.assertEqual(restarted_client.get("/api/concepts").json[0]["name"], "Plurality")

    def test_missing_records_return_json_404(self):
        response = self.client.get("/api/thinkers/999")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json, {"error": "Thinker not found."})


if __name__ == "__main__":
    unittest.main()
