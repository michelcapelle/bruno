# bruno

A genealogy to military strategic thought: a small Python web app for following ideas and thinkers.

## Run with Docker Compose

```powershell
docker compose up --build
```

Open <http://127.0.0.1:5000> in your browser. The API and MongoDB start together; MongoDB is available on `127.0.0.1:27017`. The browser/API port can be changed with `BRUNO_PORT`, and the MongoDB host port with
`BRUNO_MONGODB_PORT`.

Thinker metadata remains in `instance/bruno.sqlite3`. Wikipedia HTML and DBpedia JSON are stored as documents in MongoDB, persisted in the `mongodb-data` volume, and served by the API. The thinker card links to the plain-text Wikipedia extract rather than the saved HTML page. No source-document copies are written to the project directory. Use `docker compose down` to stop the services; the MongoDB volume is preserved. To remove the MongoDB data deliberately, use `docker compose down --volumes`.

## Run locally

Install MongoDB separately and make it available at `mongodb://localhost:27017`, then install and run Bruno:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
py -m nltk.downloader punkt_tab
bruno
```

Set `BRUNO_MONGODB_URI` and optionally `BRUNO_MONGODB_DATABASE` to configure a different MongoDB connection. The web app listens on `127.0.0.1:5000` when run through the `bruno` command.

## API

The JSON API is available at `/api/concepts` and `/api/thinkers`. The topic collection retains the `/api/concepts` route and JSON field names for compatibility. Both resources support `GET` and `POST` on the collection URL and `GET`, `PATCH`, and `DELETE` on an item URL (for example, `/api/concepts/1`).

Topics have a required `name` and optional `description`. Thinker processing requires a Wikipedia URL in `biography`. The saved thinker name comes from the matching Wikipedia-based DBpedia record, with birth/death dates and an active period when available. Use `POST /api/thinkers/process` to download the Wikipedia page, retrieve its plain text through Wikipedia's MediaWiki API, and download matching DBpedia JSON. DBpedia downloads allow up to two minutes for slower responses. The local language model defaults to the prompt “What are the thinker's contributions to military strategic thought?” and will answer using available source texts and its foundational knowledge, without mentioning the thinker by name, as a concise answer of at most 500 words. The model request allows up to 15 minutes and keeps the model loaded during processing.  The source pages, extracted text, DBpedia data, and model response are stored in MongoDB and available at `/api/documents/<document-id>`; the thinker card links to the Wikipedia plain-text extract, followed by `DBpedia` and `Ollama`. Use `PATCH /api/thinkers/<id>/process` to update an existing thinker the same way. Processing an article that is already in the collection replaces its existing thinker and source documents instead of creating another entry. Requests use JSON. Invalid input returns `400`; missing items return `404`.

### BERTopic topics

Compose starts a single Ollama container, which serves requests and pulls `qwen2.5:3b`, an English-capable open-weight model released under the Apache 2.0 license. Its weights persist in the `ollama-data` volume. The first startup downloads the model. Ollama is used for military-strategy analysis; topic generation uses BERTopic locally. Set `BRUNO_LLM_MODEL` to choose another Ollama model. The UI's connection indicator checks Ollama's model list instead of running inference, so periodic health checks do not compete with thinker processing.

Choose **Wikipedia text** (the default) or **Ollama analysis** in the source selector, then click **Generate topics** to run BERTopic over that source for every thinker. Following BERTopic's sentence-splitting best practice, Bruno splits each source into sentences before modeling and carries the originating thinker ID and name along with every sentence. Topic membership and probabilities are then combined per thinker and topic, while the saved original texts remain intact for the approximate-distribution visualization. Wikipedia HTML is not used. If an older thinker has no saved Wikipedia text extract, the app retrieves plain text from Wikipedia's MediaWiki API and caches it before topic modeling. Ollama analyses must already exist; reprocess a thinker if one is missing. Each non-empty topic is saved with its key terms and contributing thinkers. If a selected source cannot be retrieved, generation stops with an error rather than analyzing an incomplete corpus.

The Thinkers tab includes separate **What is DBpedia?** and **What is Ollama?** explanations below the thinker collection. The Topics tab includes **What is BERTopic?** below the topic-generation form. Topic generation always uses English stop words, a frequent-words threshold of `min_df=3`, and a maximum of three words per topic by default (vectorizer `ngram_range=(1, 3)`). The threshold and maximum word count can be adjusted in the form, as can **Top n words** (BERTopic `top_n_words`, default 15) and **Diversity** (default 0.2). Topic representations are always diversified with BERTopic's `MaximalMarginalRelevance`.

Topics are shown in ascending zero-based topic-number order. Each topic assignment shows only its Gaussian-mixture posterior probability and links to the thinker's Ollama analysis. The Thinkers collection includes a live search filter.

The Visualizations tab includes BERTopic's intertopic distance map, topic similarity heatmap, topics over time, topics per thinker, hierarchical clustering, document visualization, and approximate topic distributions. Topics per thinker uses every sentence from a thinker's saved source text and the thinker name as its class. Topics over time uses each thinker's approximate year (the midpoint of their active period when available, otherwise their lifespan midpoint or birth year) as the timestamp for each sentence from that thinker. For readability, all approximate years before 1800 are displayed as 1800, and the chart notes this adjustment.

When no topic count is specified, Bruno uses `max(1, min(sentence_count, round(sqrt(sentence_count / 2))))`, where each sentence is a modeling document. An explicit topic count must be no greater than the number of thinker source texts.

In the **Topics** tab, search for a phrase with BERTopic's `find_topics()`. Bruno returns as many related topics as are currently generated. Results include each topic's representative terms and similarity score. For API compatibility, this operation remains available at `POST /api/concepts/find-topics` with `{"concept": "military logistics"}`.

BERTopic models are saved in `instance/bertopic_model` after generation and loaded to create the **Visualizations** tab and power topic search. The tab offers an intertopic distance map (`visualize_topics()`), topic similarity heatmap (`visualize_heatmap()`), topics over time (`topics_over_time()`), topics per thinker (`visualize_topics_per_class()`), approximate word distributions (`visualize_approximate_distribution()`), hierarchical clustering (`visualize_hierarchy()`), and document visualization. Topics per thinker uses `topics_per_class()` to treat each thinker's source text as a separate class. The approximate distributions are shown as a topic legend above the selected visualization. Click a topic to show its key terms and the five thinkers with the highest approximate distribution percentages. Bruno saves the exact source texts and thinker names beside the model so the distribution chart uses the same Ollama analyses or Wikipedia texts that generated the topics, and displays the token-level distribution for each thinker. Topic distributions are calculated for all saved source texts in one batch. Topics over time uses the approximate years recorded for each thinker's lifespan or active period. A later generation updates the saved model and source texts. The intertopic map needs at least three topics and hierarchical clustering at least two; the tab explains these minimums. Regenerate a model created before topic search or approximate-distribution support was added to create its supporting files.

The Ollama prompt asks the model to describe military strategic ideas without naming the thinker or other people, places, organizations, battles, countries, years, or dates. It refers to the thinker only as "he" or "his".

Call `POST /api/concepts/generate` with an empty JSON object to use Wikipedia texts, or set `topic_source` to `wikipedia` or `ollama`. For compatibility, the response still returns the topic list in the `concepts` field, alongside the `BERTopic` algorithm name, the selected `topic_source`, and the number of thinker texts analyzed. Requests without thinkers or with a missing selected source return an error without generating partial results.

## Run tests

```powershell
py -m unittest discover -s tests
```
