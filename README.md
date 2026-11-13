# Bruno

Bruno is a small Python web app for exploring the genealogy of military strategic thought. It collects thinkers and their source texts, uses a local language model to analyze them, and uses BERTopic to discover and visualize themes across the collection.

## What Bruno does

- Manage thinkers from Wikipedia pages or manually entered text.
- Retrieve Wikipedia source text and matching DBpedia data.
- Generate and save an Ollama analysis for each thinker.
- Discover topics from source texts or Ollama analyses with BERTopic.
- Search for related topics and explore interactive visualizations.
- Expose the collection and its documents through a JSON API.

Thinker metadata is stored in SQLite. Wikipedia HTML, source text, DBpedia data, and model responses are stored in MongoDB. BERTopic models and their supporting files are saved under `instance/bertopic_model`.

## Quick start with Docker Compose

Docker Compose starts the Flask API, MongoDB, and Ollama:

```powershell
docker compose up --build
```

Open <http://127.0.0.1:5000>. On first startup, Ollama downloads the configured model (by default `qwen2.5:3b`).

To stop the services, run:

```powershell
docker compose down
```

The `mongodb-data` and `ollama-data` volumes are preserved. To deliberately remove the MongoDB and Ollama data as well:

```powershell
docker compose down --volumes
```

The API uses `./instance` for SQLite data and `./data` for exports on the host. The **Export** tab clears the export directory, writes the current thinkers, topics, and API status as JSON, and copies referenced source and analysis documents to `data/documents`.

## Run locally

Install MongoDB separately and make it available at `mongodb://localhost:27017`. Then create a virtual environment and install Bruno:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
py -m nltk.downloader punkt_tab
bruno
```

The app listens on `127.0.0.1:5000` when started with `bruno`.

## Using the app

### Thinkers and analyses

Add a thinker using a Wikipedia URL or manual text. The Wikipedia flow downloads the page, retrieves plain text through Wikipedia's MediaWiki API, and fetches matching DBpedia JSON. Manual entries are saved as editable `Text` documents and do not use Wikipedia or DBpedia. Processing can take time: DBpedia requests may take up to two minutes, and Ollama requests up to 15 minutes.

Bruno saves the source material and analysis in MongoDB and serves documents at `/api/documents/<document-id>`. A thinker card links to its `Wikipedia` or `Text` source, `DBpedia` when available, and `Ollama` analysis. Processing an article already in the collection replaces the existing thinker and its source documents.

### Topics and visualizations

In the **Topics** tab, choose **(Wikipedia) text** (the default, including manually entered text) or **Ollama analysis** as the source, then select **Generate topics**. Ollama analyses must already exist for all selected thinkers. Bruno splits source texts into sentences before modeling and retains each sentence's thinker association. If a selected source is missing, generation stops with an error rather than modeling an incomplete corpus.

Seed words are enabled by default; edit the suggested military-strategy vocabulary or turn it off before generating. Other options include the number of topics, the frequent-words threshold, maximum words per topic, **Top n words**, and **Diversity**. By default Bruno uses English stop words, `min_df=3`, up to three words per topic, `top_n_words=15`, and a diversity of `0.2`. Topic representations use BERTopic's `MaximalMarginalRelevance`. If no topic count is specified, Bruno uses `max(1, min(sentence_count, round(sqrt(sentence_count / 2))))`; an explicit count cannot exceed the number of thinker source texts.

Generated topics are saved with their key terms and contributing thinkers. Topics are shown in ascending zero-based topic-number order. Each assignment shows its Gaussian-mixture posterior probability. Use topic search to find related topics and see their representative terms and similarity scores.

The **Visualizations** tab provides an intertopic distance map, similarity heatmap, topics over time, topics per thinker, hierarchical clustering, document visualization, and approximate topic distributions. Approximate distributions are calculated from the saved source texts used to generate the model. Select a topic to see its key terms and the five thinkers with the highest approximate distribution percentages. Topics over time uses each thinker's approximate year: the active-period midpoint when available, otherwise the lifespan midpoint or birth year. Years before 1800 are displayed as 1800. The intertopic map requires at least three topics and hierarchical clustering requires at least two.

Models and their source texts are saved under `instance/bertopic_model`. Regenerate a model created before topic search or approximate-distribution support was added to create the required supporting files.

## API

Requests and responses use JSON. The API index is at `/api`.

| Resource | Collection operations | Item operations |
| --- | --- | --- |
| Topics (legacy route name: concepts) | `GET`, `POST /api/concepts` | `GET`, `PATCH`, `DELETE /api/concepts/<id>` |
| Thinkers | `GET`, `POST /api/thinkers` | `GET`, `PATCH`, `DELETE /api/thinkers/<id>` |
| Documents | — | `GET`, `PUT /api/documents/<document-id>` |

Topic generation is available at `POST /api/concepts/generate`; topic search is at `POST /api/concepts/find-topics`. The concepts route and response field names are retained for compatibility.

### Topic generation

Send an empty JSON object to use the default source (Wikipedia or manually entered text):

```json
{}
```

Set `topic_source` to `wikipedia` or `ollama` to choose a source. The optional `use_seed_topics` boolean and `seed_topics` string list configure seed words. The response includes the topics in the `concepts` field, the algorithm name, selected source, and number of thinker texts analyzed. Requests without thinkers or with a missing selected source return an error without generating partial results.

### Topic search

Send a phrase to `POST /api/concepts/find-topics`:

```json
{
  "concept": "military logistics"
}
```

Bruno returns related generated topics with their representative terms and similarity scores.

### Thinker processing

Thinker creation accepts either a Wikipedia URL in `biography`, or a manual `name` and `text`. Manual entries may also include `birth_year`, `death_year`, `active_start_year`, and `active_end_year`. Use `PATCH /api/thinkers/<id>/process` to process or update an existing thinker. Invalid input returns `400`; missing items return `404`.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `BRUNO_PORT` | Host port for the API in Docker Compose | `5000` |
| `BRUNO_MONGODB_PORT` | Host port for MongoDB in Docker Compose | `27017` |
| `BRUNO_MONGODB_URI` | MongoDB connection URI | `mongodb://localhost:27017` |
| `BRUNO_MONGODB_DATABASE` | MongoDB database name | `bruno` |
| `BRUNO_DATABASE` | Path to the SQLite database | `instance/bruno.sqlite3` |
| `BRUNO_DATA_PATH` | Export directory | `data` |
| `BRUNO_LLM_MODEL` | Ollama model used for analysis | `qwen2.5:3b` |
| `BRUNO_LLM_URL` | Ollama generate API URL | `http://localhost:11434/api/generate` |
| `BRUNO_EMBEDDING_MODEL` | Sentence-transformer model used by BERTopic | Application default |
| `BRUNO_API_BASE_URL` | Public API origin for a Firebase-hosted UI | Empty (same-origin) |
| `BRUNO_ALLOWED_ORIGINS` | Comma-separated CORS origins allowed by the API | Empty |

## Tests

Run the test suite with:

```powershell
py -m unittest discover -s tests
```
