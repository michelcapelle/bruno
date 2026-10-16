"""Web application and JSON API for concepts and thinkers."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

EntitySpec = dict[str, Any]

ENTITIES: dict[str, EntitySpec] = {
    "concepts": {
        "fields": {"name": ("TEXT", True), "description": ("TEXT", False)},
        "label": "Concept",
    },
    "thinkers": {
        "fields": {"name": ("TEXT", True), "biography": ("TEXT", False)},
        "label": "Thinker",
    },
}


def create_app(database_path: str | Path | None = None) -> Flask:
    """Create the Flask application and initialize its SQLite database."""
    app = Flask(__name__)
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    db_path = Path(
        database_path
        or os.environ.get("BRUNO_DATABASE", instance_path / "bruno.sqlite3")
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.config["DATABASE"] = str(db_path)
    _initialize_database(app.config["DATABASE"])

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api")
    def api_index():
        return jsonify(
            {
                "concepts": "/api/concepts",
                "thinkers": "/api/thinkers",
            }
        )

    for entity, spec in ENTITIES.items():
        _register_entity_routes(app, entity, spec)

    return app


def _connect(database_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _database(database_path: str) -> Generator[sqlite3.Connection, None, None]:
    connection = _connect(database_path)
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_database(database_path: str) -> None:
    with _database(database_path) as connection:
        for entity, spec in ENTITIES.items():
            columns = ", ".join(
                f'"{field}" {field_type}'
                + (" NOT NULL" if required else "")
                for field, (field_type, required) in spec["fields"].items()
            )
            connection.execute(
                f'CREATE TABLE IF NOT EXISTS "{entity}" '
                f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {columns})"
            )


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _validate_payload(
    entity: str, payload: Any, *, partial: bool
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    fields = ENTITIES[entity]["fields"]
    unknown_fields = payload.keys() - fields.keys()
    if unknown_fields:
        return None, f"Unknown field: {sorted(unknown_fields)[0]}."
    if not partial and "name" not in payload:
        return None, "The name field is required."

    validated: dict[str, str] = {}
    for field, value in payload.items():
        if not isinstance(value, str):
            return None, f"The {field} field must be a string."
        value = value.strip()
        if field == "name" and not value:
            return None, "The name field cannot be empty."
        max_length = 200 if field == "name" else 10000
        if len(value) > max_length:
            return None, f"The {field} field must be at most {max_length} characters."
        validated[field] = value
    return validated, None


def _register_entity_routes(
    app: Flask, entity: str, spec: EntitySpec
) -> None:
    table = entity

    def collection():
        with _database(app.config["DATABASE"]) as connection:
            rows = connection.execute(
                f'SELECT * FROM "{table}" ORDER BY id DESC'
            ).fetchall()
        return jsonify([_serialize(row) for row in rows])

    def create():
        payload, error = _validate_payload(entity, request.get_json(silent=True), partial=False)
        if error:
            return jsonify({"error": error}), 400
        assert payload is not None
        insert_fields = list(payload)
        columns = ", ".join(f'"{field}"' for field in insert_fields)
        placeholders = ", ".join("?" for _ in insert_fields)
        with _database(app.config["DATABASE"]) as connection:
            cursor = connection.execute(
                f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})',
                [payload[field] for field in insert_fields],
            )
            row = connection.execute(
                f'SELECT * FROM "{table}" WHERE id = ?', (cursor.lastrowid,)
            ).fetchone()
        return jsonify(_serialize(row)), 201

    def item(item_id: int):
        with _database(app.config["DATABASE"]) as connection:
            row = connection.execute(
                f'SELECT * FROM "{table}" WHERE id = ?', (item_id,)
            ).fetchone()
        if row is None:
            return jsonify({"error": f'{spec["label"]} not found.'}), 404
        return jsonify(_serialize(row))

    def update(item_id: int):
        payload, error = _validate_payload(entity, request.get_json(silent=True), partial=True)
        if error:
            return jsonify({"error": error}), 400
        assert payload is not None
        if not payload:
            return jsonify({"error": "Provide at least one field to update."}), 400
        assignments = ", ".join(f'"{field}" = ?' for field in payload)
        with _database(app.config["DATABASE"]) as connection:
            cursor = connection.execute(
                f'UPDATE "{table}" SET {assignments} WHERE id = ?',
                [*payload.values(), item_id],
            )
            if cursor.rowcount == 0:
                return jsonify({"error": f'{spec["label"]} not found.'}), 404
            row = connection.execute(
                f'SELECT * FROM "{table}" WHERE id = ?', (item_id,)
            ).fetchone()
        return jsonify(_serialize(row))

    def delete(item_id: int):
        with _database(app.config["DATABASE"]) as connection:
            cursor = connection.execute(
                f'DELETE FROM "{table}" WHERE id = ?', (item_id,)
            )
        if cursor.rowcount == 0:
            return jsonify({"error": f'{spec["label"]} not found.'}), 404
        return "", 204

    collection.__name__ = f"list_{entity}"
    create.__name__ = f"create_{entity}"
    item.__name__ = f"get_{entity}_item"
    update.__name__ = f"update_{entity}"
    delete.__name__ = f"delete_{entity}"
    app.add_url_rule(f"/api/{entity}", view_func=collection, methods=["GET"])
    app.add_url_rule(f"/api/{entity}", view_func=create, methods=["POST"])
    app.add_url_rule(
        f"/api/{entity}/<int:item_id>", view_func=item, methods=["GET"]
    )
    app.add_url_rule(
        f"/api/{entity}/<int:item_id>", view_func=update, methods=["PATCH"]
    )
    app.add_url_rule(
        f"/api/{entity}/<int:item_id>", view_func=delete, methods=["DELETE"]
    )


def main() -> None:
    """Run the development server."""
    create_app().run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
