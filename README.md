# bruno

Genealogies of Conceptual Thinking: a small Python web app for managing
concepts and thinkers.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
bruno
```

Open <http://127.0.0.1:5000> in your browser. Data is stored in
`instance/bruno.sqlite3`.

## API

The JSON API is available at `/api/concepts` and `/api/thinkers`. Both
resources support `GET` and `POST` on the collection URL and `GET`, `PATCH`,
and `DELETE` on an item URL (for example, `/api/concepts/1`).

Concepts have a required `name` and optional `description`. Thinkers have a
required `name` and optional `biography`. Create and update requests use JSON.
Invalid input returns `400`; missing items return `404`.

## Run tests

```powershell
py -m unittest discover -s tests
```
