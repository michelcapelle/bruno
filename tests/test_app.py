import json
import io
import sqlite3
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock, patch

from bson import ObjectId
from pymongo.errors import PyMongoError

from bruno.app import (
    _download_dbpedia_data,
    _download_wikipedia_text,
    _approximate_distributions,
    _concepts_from_distributions,
    _dbpedia_year,
    _extract_military_contributions,
    _load_bertopic_model,
    _load_topic_source_documents,
    _save_topic_search_vectorizer,
    _save_topic_source_documents,
    _split_documents_into_sentences,
    create_app,
)


class FakeMongoClient:
    def __init__(self):
        self.documents = FakeMongoCollection()

    def __getitem__(self, name):
        return self if name != "documents" else self.documents


class FakeMongoCollection:
    def __init__(self):
        self.records = {}
        self.fail_on_source = None

    def insert_one(self, document):
        if document["source"] == self.fail_on_source:
            raise PyMongoError("injected MongoDB failure")
        document_id = ObjectId()
        self.records[str(document_id)] = {**document, "_id": document_id}
        return SimpleNamespace(inserted_id=document_id)

    def find_one(self, query):
        if "_id" in query:
            return self.records.get(str(query["_id"]))
        for document in self.records.values():
            if all(document.get(key) == value for key, value in query.items()):
                return document
        return None

    def delete_many(self, query):
        if "_id" in query:
            document_ids = {str(value) for value in query["_id"]["$in"]}
            removed = [
                key for key in self.records if key in document_ids
            ]
        else:
            removed = [
                key
                for key, document in self.records.items()
                if all(document.get(field) == value for field, value in query.items())
            ]
        for key in removed:
            del self.records[key]
        return SimpleNamespace(deleted_count=len(removed))


def dbpedia_person(
    slug,
    name,
    birth_date="1780-07-01",
    death_date="1831-11-16",
    service_start="1792",
    service_end="1831",
):
    def literal(value):
        return [{"type": "literal", "value": value, "lang": "en"}]

    properties = {
        "http://dbpedia.org/property/name": literal(name),
        "http://dbpedia.org/ontology/birthDate": literal(birth_date),
        "http://dbpedia.org/ontology/deathDate": literal(death_date),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": [
            {"type": "uri", "value": "http://dbpedia.org/ontology/Person"},
        ],
        "http://purl.org/dc/terms/subject": [
            {"type": "uri", "value": "http://dbpedia.org/resource/Category:Military_theorists"},
        ],
    }
    if service_start:
        properties["http://dbpedia.org/ontology/serviceStartYear"] = literal(service_start)
    if service_end:
        properties["http://dbpedia.org/ontology/serviceEndYear"] = literal(service_end)
    return json.dumps(
        {f"http://dbpedia.org/resource/{slug}": properties}
    ).encode("utf-8")


class BrunoApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.mongo_client = FakeMongoClient()
        self.app = create_app(database_path, mongo_client=self.mongo_client)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        model_patcher = patch(
            "bruno.app._extract_military_contributions",
            return_value="Clausewitz developed influential ideas on military strategy.",
        )
        self.extract_military_contributions = model_patcher.start()
        self.addCleanup(model_patcher.stop)
        text_patcher = patch(
            "bruno.app._download_wikipedia_text",
            return_value="Clausewitz developed influential ideas on war.",
        )
        self.download_wikipedia_text = text_patcher.start()
        self.addCleanup(text_patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_thinker_text(self, name, text):
        document_key = f"wiki-text-{name.lower().replace(' ', '-')}"
        wikipedia_result = self.mongo_client.documents.insert_one(
            {
                "document_key": document_key,
                "source": "wikipedia-text",
                "url": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                "content_type": "text/plain; charset=utf-8",
                "content": text.encode("utf-8"),
            }
        )
        ollama_result = self.mongo_client.documents.insert_one(
            {
                "document_key": document_key,
                "source": "military-strategy",
                "url": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                "content_type": "text/plain; charset=utf-8",
                "content": f"OLLAMA analysis: {text}".encode("utf-8"),
            }
        )
        connection = sqlite3.connect(self.app.config["DATABASE"])
        try:
            cursor = connection.execute(
                "INSERT INTO thinkers "
                "(name, biography, wiki_document_key, wiki_text_document_id, "
                "military_thinking_document_id) VALUES (?, ?, ?, ?, ?)",
                (
                    name,
                    f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                    document_key,
                    str(wikipedia_result.inserted_id),
                    str(ollama_result.inserted_id),
                ),
            )
            connection.commit()
            return cursor.lastrowid
        finally:
            connection.close()

    def test_homepage_and_api_index_are_available(self):
        homepage = self.client.get("/")
        api_index = self.client.get("/api")
        app_script = self.client.get("/static/app.js")

        self.assertEqual(homepage.status_code, 200)
        self.assertEqual(app_script.status_code, 200)
        self.assertEqual(app_script.data.count(b'target = "_blank"'), 5)
        self.assertEqual(app_script.data.count(b'rel = "noopener noreferrer"'), 5)
        app_script.close()
        self.assertIn(b"Assemblies of Conceptual Thinking", homepage.data)
        self.assertIn(b"A GENEALOGY TO MILITARY STRATEGIC THOUGHT", homepage.data)
        self.assertNotIn(b"A FIELD GUIDE TO IDEAS", homepage.data)
        self.assertEqual(homepage.data.count(b"<blockquote>"), 1)
        self.assertIn(
            b"The task of defining and ordering the social should be left to the actors themselves, not taken up by the analyst",
            homepage.data,
        )
        self.assertIn(
            b'<a href="https://en.wikipedia.org/wiki/List_of_military_theorists_and_writers" target="_blank" rel="noopener noreferrer">',
            homepage.data,
        )
        self.assertLess(
            homepage.data.index(b'id="entry-wikipedia"'),
            homepage.data.index(b"Browse Wikipedia military theorists and writers"),
        )
        self.assertNotIn(
            b"<p>Assemblies of Conceptual Thinking</p>",
            homepage.data,
        )
        self.assertLess(
            homepage.data.index(b'data-collection="thinkers"'),
            homepage.data.index(b'data-collection="concepts"'),
        )
        self.assertIn(
            b'data-collection="about" aria-selected="true"',
            homepage.data,
        )
        self.assertIn(b'data-collection="visualizations"', homepage.data)
        self.assertIn(b'id="visualization-count"', homepage.data)
        self.assertIn(b'aria-label="8 available visualizations">8</span>', homepage.data)
        self.assertIn(b'id="topic-distribution-retry"', homepage.data)
        self.assertIn(b'id="topic-distribution-bubbles"', homepage.data)
        self.assertIn(b'sandbox="allow-scripts allow-downloads"', homepage.data)
        self.assertIn(
            b'Click a topic to see its key terms and the five thinkers with the highest percentages.',
            homepage.data,
        )
        self.assertIn(b'data-visualization="topics-over-time" aria-pressed="false">Over Time', homepage.data)
        self.assertIn(b'data-visualization="topics-per-class" aria-pressed="false">Per Thinker', homepage.data)
        self.assertIn(b'role="group" aria-label="Choose a visualization"', homepage.data)
        self.assertIn(b'data-visualization="topics" aria-pressed="true"', homepage.data)
        self.assertIn(b'data-visualization="heatmap" aria-pressed="false">Similarity Heatmap', homepage.data)
        self.assertIn(b'data-visualization="approximate-distribution" aria-pressed="false"', homepage.data)
        self.assertIn(b'id="visualization-loading"', homepage.data)
        self.assertIn(
            b'data-visualization="approximate-distribution" aria-pressed="false">Approximate Distributions',
            homepage.data,
        )
        self.assertIn(b'id="topics-visualization"', homepage.data)
        self.assertIn(
            b"topicsVisualization.srcdoc = html",
            app_script.data,
        )
        self.assertIn(
            b'fetch(`/visualizations/${visualization}`',
            app_script.data,
        )
        self.assertIn(
            b"Calculating approximate distributions for the saved source texts",
            app_script.data,
        )
        self.assertIn(
            b'option.dataset.visualization === "approximate-distribution"',
            app_script.data,
        )
        self.assertIn(b'id="entry-name"', homepage.data)
        self.assertIn(b'id="entry-name" name="name"', homepage.data)
        self.assertIn(b'hidden>Name</label>', homepage.data)
        self.assertIn(b"military_thinking_document_id", app_script.data)
        self.assertIn(b"wiki_text_document_id", app_script.data)
        self.assertIn(b"String(topicId)", app_script.data)
        self.assertNotIn(b"String(topicId).padStart", app_script.data)
        visualization_markup = homepage.data.split(
            b'<section class="visualizations"',
            1,
        )[1]
        self.assertNotIn(b"BERTopic", visualization_markup)
        self.assertIn(b">Intertopic Distance<", visualization_markup)
        self.assertNotIn(b">Topic map<", visualization_markup)
        self.assertIn(b'reprocessButton.textContent = "Reprocess"', app_script.data)
        self.assertIn(b'if (activeCollection === "thinkers")', app_script.data)
        self.assertIn(b'`/api/thinkers/${entry.id}/process`', app_script.data)
        self.assertIn(b"prompt: ollamaPromptInput.value", app_script.data)
        self.assertIn(
            b"word_count: Number(responseWordCountInput.value)",
            app_script.data,
        )
        self.assertEqual(app_script.data.count(b"/documents/${encodeURIComponent("), 5)
        self.assertLess(
            app_script.data.index(b'articleText.textContent = "Wikipedia"'),
            app_script.data.index(b'dbpediaPage.textContent = "DBpedia"'),
        )
        self.assertLess(
            app_script.data.index(b'dbpediaPage.textContent = "DBpedia"'),
            app_script.data.index(b'militaryThinking.textContent = "Ollama"'),
        )
        self.assertNotIn(b'savedPage.textContent = "Wikipedia"', app_script.data)
        self.assertIn(b"id=\"language-model-status\"", homepage.data)
        self.assertIn(b'id="form-status"', homepage.data)
        self.assertIn(
            b"Ollama LLM answer length (words)",
            homepage.data,
        )
        self.assertIn(
            b'id="response-word-count" type="number" min="100" max="1000" step="1" value="500"',
            homepage.data,
        )
        self.assertIn(
            b'id="ollama-prompt" rows="4" maxlength="5000"',
            homepage.data,
        )
        self.assertIn(b">Ollama LLM prompt</label>", homepage.data)
        self.assertIn(
            b">What are the thinker's contributions to military strategic thought?</textarea>",
            homepage.data,
        )
        self.assertNotIn(
            b"Refer to the thinker only as he or his; do not mention his name.</textarea>",
            homepage.data,
        )
        self.assertIn(
            b"What are the thinker's contributions to military strategic thought?",
            app_script.data,
        )
        self.assertIn(b"payload.prompt = ollamaPromptInput.value", app_script.data)
        self.assertLess(
            homepage.data.index(b'id="ollama-prompt-label"'),
            homepage.data.index(b'id="response-word-count-label"'),
        )
        self.assertIn(b"Trace the ideas of influential strategists", homepage.data)
        self.assertNotIn(
            b"Keep concepts and the thinkers behind them in one thoughtful, searchable place.",
            homepage.data,
        )
        self.assertIn(b'id="topic-source" hidden', homepage.data)
        self.assertIn(b'<option value="ollama">Ollama analysis</option>', homepage.data)
        self.assertIn(b'<option value="wikipedia" selected>Wikipedia text</option>', homepage.data)
        self.assertIn(b'payload.topic_source = topicSourceSelect.value', app_script.data)
        self.assertNotIn(b'topicSourceSelect.value = "wikipedia"', app_script.data)
        self.assertIn(b"async function loadCollectionCounts()", app_script.data)
        self.assertIn(b'requestJson("/api/thinkers")', app_script.data)
        self.assertIn(b'requestJson("/api/concepts")', app_script.data)
        self.assertIn(b"loadCollectionCounts();", app_script.data)
        self.assertIn(
            b"if (showVisualizations) {\n      loadVisualization();\n      loadTopicDistributions();",
            app_script.data.replace(b"\r\n", b"\n"),
        )
        self.assertIn(b"payload.min_df = Number(topicMinDfInput.value)", app_script.data)
        self.assertIn(b"payload.max_words_in_topic = Number(topicMaxWordsInput.value)", app_script.data)
        self.assertNotIn(b'id="topic-stop-words"', homepage.data)
        self.assertIn(b">Word frequency threshold</label>", homepage.data)
        self.assertIn(b'id="topic-max-words" type="number" min="1" step="1" value="3"', homepage.data)
        self.assertIn(b">Compound words</label>", homepage.data)
        self.assertIn(b">Top n words</label>", homepage.data)
        self.assertIn(b">Diversity</label>", homepage.data)
        self.assertIn(b'id="topic-diversity" type="number" min="0" max="1" step="0.1" value="0.2"', homepage.data)
        self.assertIn(b"payload.diversity = Number(topicDiversityInput.value)", app_script.data)
        self.assertIn(b'id="topic-top-n-words" type="number" min="1" step="1" value="15"', homepage.data)
        self.assertIn(b"payload.top_n_words = Number(topicTopNWordsInput.value)", app_script.data)
        self.assertLess(
            homepage.data.index(b'id="entry-wikipedia"'),
            homepage.data.index(b'id="wikipedia-suggestions"'),
        )
        self.assertIn(
            b"payload.word_count = Number(responseWordCountInput.value)",
            app_script.data,
        )
        self.assertNotIn(b"Topic assignments", app_script.data)
        self.assertIn(b"assignment.probability * 100", app_script.data)
        self.assertIn(b'showTopicDistributionDetails(topic, bubble)', app_script.data)
        self.assertIn(b'addEventListener("click"', app_script.data)
        self.assertIn(b'.slice(0, 5)', app_script.data)
        self.assertIn(b'setAttribute("aria-expanded", "false")', app_script.data)
        self.assertNotIn(b'addEventListener("pointerenter"', app_script.data)
        self.assertNotIn(b'addEventListener("focus"', app_script.data)
        self.assertNotIn(b"Topic ${String(assignment.topic_id + 1)", app_script.data)
        self.assertIn(b'data-collection="visualizations"', homepage.data)
        self.assertIn(b"What is DBpedia?", homepage.data)
        self.assertIn(b"What is Ollama?", homepage.data)
        self.assertIn(b"What is BERTopic?", homepage.data)
        self.assertIn(
            b'href="https://maartengr.github.io/BERTopic/" target="_blank" rel="noopener noreferrer">Learn more about BERTopic</a>',
            homepage.data,
        )
        self.assertIn(b'id="topic-information"', homepage.data)
        self.assertIn(b'id="thinker-information"', homepage.data)
        self.assertIn(b'class="document-section document-disclosure info-panel bertopic-info"', homepage.data)
        self.assertIn(b'<div class="ollama-info-content">', homepage.data)
        self.assertIn(b'id="thinker-search" type="search"', homepage.data)
        self.assertIn(b'id="topic-count" type="number" min="1" step="1"', homepage.data)
        self.assertIn(b"Number of topics (optional)", homepage.data)
        self.assertIn(b'id="topic-model-options" hidden', homepage.data)
        self.assertNotIn(b'id="topic-stop-words"', homepage.data)
        self.assertIn(b'id="topic-min-df" type="number" min="1" step="1" value="3"', homepage.data)
        self.assertNotIn(b'id="topic-ngram-range"', homepage.data)
        self.assertIn(b'id="topic-search-box"', homepage.data)
        self.assertIn(b'id="topic-search-concept" type="text"', homepage.data)
        self.assertIn(b"Find related topics", homepage.data)
        self.assertIn(b">Topic</label>", homepage.data)
        self.assertIn(b">Find topics <", homepage.data)
        self.assertLess(
            homepage.data.index(b'<section class="editor"'),
            homepage.data.index(b'id="topic-search-heading"'),
        )
        self.assertLess(
            homepage.data.index(b'id="topic-search-heading"'),
            homepage.data.index(b'<section class="entries"'),
        )
        self.assertLess(
            homepage.data.index(b'id="topic-search-heading"'),
            homepage.data.index(b'id="topic-information"'),
        )
        self.assertLess(
            homepage.data.index(b'id="topic-information"'),
            homepage.data.index(b'<section class="visualizations"'),
        )
        self.assertLess(
            homepage.data.index(b'<section class="editor"'),
            homepage.data.index(b'id="thinker-information"'),
        )
        self.assertLess(
            homepage.data.index(b'id="thinker-information"'),
            homepage.data.index(b'id="topic-search-heading"'),
        )
        self.assertIn(b'data-visualization="hierarchy"', homepage.data)
        self.assertIn(b"Hierarchical Clustering", homepage.data)
        self.assertNotIn(b"Concept (topic)", homepage.data)
        self.assertNotIn(b">top_n</label>", homepage.data)
        self.assertNotIn(b'id="topic-search-top-n"', homepage.data)
        self.assertIn(b'"/api/concepts/find-topics"', app_script.data)
        self.assertIn(b"String(result.topic_id)", app_script.data)
        self.assertIn(b"Math.round(result.similarity * 100)", app_script.data)
        self.assertIn(b"similarityPercent > 0", app_script.data)
        self.assertIn(
            b"(result) => Math.round(result.similarity * 100) > 0",
            app_script.data,
        )
        self.assertIn(b"String(result.topic_id)", app_script.data)
        self.assertNotIn(b"String(result.topic_id).padStart", app_script.data)
        self.assertIn(b'topicNumber.className = "entry-index"', app_script.data)
        self.assertIn(b"conceptCount < 1", app_script.data)
        self.assertNotIn(b"Topic ${String(result.topic_id", app_script.data)
        self.assertNotIn(
            b"Discover topics from one text source for all thinkers.",
            homepage.data,
        )
        self.assertIn(b"payload.topic_count = Number(topicCountInput.value)", app_script.data)
        self.assertIn(b"payload.min_df = Number(topicMinDfInput.value)", app_script.data)
        self.assertIn(b"payload.max_words_in_topic = Number(topicMaxWordsInput.value)", app_script.data)
        self.assertIn(b"Downloading sources and analyzing with the language model.", app_script.data)
        self.assertIn(b"Processing thinker", app_script.data)
        self.assertIn(b"LLM unavailable", app_script.data)
        self.assertIn(b"LLM connected", app_script.data)
        self.assertIn(b"wikipediaSuggestions.hidden = activeCollection !== \"thinkers\"", app_script.data)
        self.assertEqual(
            api_index.json,
            {
                "concepts": "/api/concepts",
                "generate_concepts": "/api/concepts/generate",
                "find_topics": "/api/concepts/find-topics",
                "thinkers": "/api/thinkers",
                "visualizations": "/visualizations/topics",
            },
        )

    @patch("bruno.app.urlopen")
    def test_language_model_status_is_connected_when_model_is_installed(
        self, open_url
    ):
        open_url.return_value = io.BytesIO(
            json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()
        )

        response = self.client.get("/api/status/language-model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"connected": True})
        self.assertEqual(
            open_url.call_args.args[0],
            "http://localhost:11434/api/tags",
        )
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 3})

    @patch("bruno.app.urlopen", side_effect=URLError("Ollama unavailable"))
    def test_language_model_status_is_disconnected_when_ollama_is_unavailable(
        self, open_url
    ):
        response = self.client.get("/api/status/language-model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"connected": False})
        open_url.assert_called_once()

    @patch("bruno.app.urlopen", side_effect=TimeoutError("timed out"))
    def test_language_model_status_is_disconnected_when_ollama_status_times_out(
        self, open_url
    ):
        response = self.client.get("/api/status/language-model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"connected": False})
        open_url.assert_called_once()

    @patch("bruno.app.urlopen")
    def test_language_model_status_is_disconnected_when_model_is_not_installed(
        self, open_url
    ):
        open_url.return_value = io.BytesIO(
            json.dumps({"models": [{"name": "other-model:latest"}]}).encode()
        )

        response = self.client.get("/api/status/language-model")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"connected": False})

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_uses_all_saved_wikipedia_text_and_persists_topics(
        self, generate_topics
    ):
        clausewitz_id = self.add_thinker_text("Clausewitz", "PLAINTEXT clausewitz")
        sun_tzu_id = self.add_thinker_text("Sun Tzu", "PLAINTEXT sun tzu")
        expected_concepts = [
            {
                "name": "war · strategy",
                "thinker_ids": [clausewitz_id, sun_tzu_id],
                "topic_assignments": [
                    {
                        "thinker_id": clausewitz_id,
                        "topic_id": 0,
                        "probability": 0.92,
                    },
                    {
                        "thinker_id": sun_tzu_id,
                        "topic_id": 0,
                        "probability": 0.75,
                    },
                ],
            },
            {
                "name": "deception · planning",
                "thinker_ids": [sun_tzu_id],
                "topic_assignments": [
                    {
                        "thinker_id": sun_tzu_id,
                        "topic_id": 1,
                        "probability": 0.84,
                    }
                ],
            },
        ]
        generate_topics.return_value = expected_concepts
        self.client.post("/api/concepts", json={"name": "Old concept"})

        response = self.client.post("/api/concepts/generate", json={})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["algorithm"], "BERTopic")
        self.assertEqual(response.json["topic_source"], "wikipedia")
        self.assertEqual(response.json["source_thinkers"], 2)
        self.assertEqual(
            [
                {key: item[key] for key in ("name", "description")}
                for item in response.json["concepts"]
            ],
            [
                {"name": "war · strategy", "description": ""},
                {"name": "deception · planning", "description": ""},
            ],
        )
        self.assertEqual(
            [thinker["name"] for thinker in response.json["concepts"][0]["thinkers"]],
            ["Clausewitz", "Sun Tzu"],
        )
        self.assertEqual(
            [thinker["name"] for thinker in response.json["concepts"][1]["thinkers"]],
            ["Sun Tzu"],
        )
        self.assertEqual(
            response.json["concepts"][0]["topic_assignments"],
            [
                {
                    "thinker_id": clausewitz_id,
                    "topic_id": 0,
                    "probability": 0.92,
                    "topic_source": "wikipedia",
                },
                {
                    "thinker_id": sun_tzu_id,
                    "topic_id": 0,
                    "probability": 0.75,
                    "topic_source": "wikipedia",
                },
            ],
        )
        self.assertEqual(
            self.client.get("/api/concepts").json[1]["topic_assignments"],
            [
                {
                    "thinker_id": clausewitz_id,
                    "topic_id": 0,
                    "probability": 0.92,
                    "topic_source": "wikipedia",
                },
                {
                    "thinker_id": sun_tzu_id,
                    "topic_id": 0,
                    "probability": 0.75,
                    "topic_source": "wikipedia",
                },
            ],
        )
        self.assertEqual(
            generate_topics.call_args.args,
            (
                ["PLAINTEXT clausewitz", "PLAINTEXT sun tzu"],
                ["Clausewitz", "Sun Tzu"],
                None,
                [clausewitz_id, sun_tzu_id],
            ),
        )
        self.assertEqual(
            generate_topics.call_args.kwargs,
            {
                "model_path": Path(self.app.config["BERTOPIC_MODEL_PATH"]),
                "stop_words": "english",
                "min_df": 3,
                "ngram_range": (1, 3),
                "top_n_words": 15,
                "diversity": 1.0,
            },
        )
        self.download_wikipedia_text.assert_not_called()
        self.assertEqual(
            [
                {key: item[key] for key in ("name", "description")}
                for item in self.client.get("/api/concepts").json
            ],
            [
                {"name": "deception · planning", "description": ""},
                {"name": "war · strategy", "description": ""},
            ],
        )
        self.assertEqual(
            [item["name"] for item in self.client.get("/api/concepts").json],
            ["deception · planning", "war · strategy"],
        )

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_passes_requested_topic_count(self, generate_topics):
        thinker_id = self.add_thinker_text("Clausewitz", "PLAINTEXT clausewitz")
        generate_topics.return_value = [
            {"name": "war", "thinker_ids": [thinker_id]}
        ]

        response = self.client.post(
            "/api/concepts/generate",
            json={"topic_count": 1, "topic_source": "ollama"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["topic_source"], "ollama")
        self.assertEqual(generate_topics.call_args.args[2], 1)
        self.assertEqual(
            generate_topics.call_args.args[0],
            ["OLLAMA analysis: PLAINTEXT clausewitz"],
        )
        self.assertEqual(
            generate_topics.call_args.kwargs["model_path"],
            Path(self.app.config["BERTOPIC_MODEL_PATH"]),
        )
        self.assertEqual(generate_topics.call_args.kwargs["stop_words"], "english")
        self.assertEqual(generate_topics.call_args.kwargs["min_df"], 3)
        self.assertEqual(generate_topics.call_args.kwargs        ["ngram_range"], (1, 3))

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_passes_custom_vectorizer_settings(
        self, generate_topics
    ):
        thinker_id = self.add_thinker_text("Clausewitz", "PLAINTEXT clausewitz")
        generate_topics.return_value = [
            {"name": "war", "thinker_ids": [thinker_id]}
        ]

        response = self.client.post(
            "/api/concepts/generate",
            json={
                "stop_words": None,
                "min_df": 2,
                "max_words_in_topic": 4,
                "top_n_words": 25,
                "diversity": 0.5,
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(generate_topics.call_args.kwargs["top_n_words"], 25)
        self.assertEqual(generate_topics.call_args.kwargs["diversity"], 0.5)
        self.assertEqual(generate_topics.call_args.kwargs["stop_words"], None)
        self.assertEqual(generate_topics.call_args.kwargs["min_df"], 2)
        self.assertEqual(generate_topics.call_args.kwargs["ngram_range"], (1, 4))

    def test_generate_concepts_rejects_invalid_vectorizer_settings(self):
        invalid_settings = (
            {"stop_words": "french"},
            {"min_df": 0},
            {"min_df": True},
            {"max_words_in_topic": 0},
            {"max_words_in_topic": True},
            {"top_n_words": 0},
            {"top_n_words": True},
            {"diversity": -0.1},
            {"diversity": 1.5},
            {"diversity": True},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                response = self.client.post(
                    "/api/concepts/generate",
                    json=settings,
                )

                self.assertEqual(response.status_code, 400)

    def test_topic_visualization_guides_user_when_no_model_has_been_saved(self):
        response = self.client.get("/visualizations/topics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Generate topics before viewing visualizations.",
            response.data,
        )

    @patch("bruno.app._load_bertopic_model")
    def test_topic_visualization_explains_minimum_topic_count(
        self, load_topic_model
    ):
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            visualize_topics=Mock(),
        )
        load_topic_model.return_value = topic_model
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()

        response = self.client.get("/visualizations/topics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"requires at least three topics", response.data)
        topic_model.visualize_topics.assert_not_called()

    @patch("bruno.app._load_bertopic_model")
    def test_topic_visualization_loads_saved_model_and_renders_plot(
        self, load_topic_model
    ):
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Rendered topic map</div>"),
        )
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: [], 2: []}),
            visualize_topics=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()

        response = self.client.get("/visualizations/topics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rendered topic map", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        load_topic_model.assert_called_once_with(model_path)
        topic_model.visualize_topics.assert_called_once_with(use_ctfidf=True)
        figure.update_layout.assert_called_once_with(width=None, autosize=True)
        figure.to_html.assert_called_once_with(
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True},
        )

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_hierarchy_visualization_renders_saved_model(
        self, split_documents, load_topic_model
    ):
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Rendered hierarchy</div>"),
        )
        hierarchical_topics = object()
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            hierarchical_topics=Mock(return_value=hierarchical_topics),
            visualize_hierarchy=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        _save_topic_source_documents(
            model_path, ["Source text"], ["First Thinker (1700-1760)"]
        )
        split_documents.return_value = (
            ["First sentence.", "Second sentence."],
            ["First Thinker (1700-1760)"] * 2,
            [0, 0],
        )
        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module

        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/hierarchy")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rendered hierarchy", response.data)
        topic_model.hierarchical_topics.assert_called_once_with(
            ["First sentence.", "Second sentence."]
        )
        topic_model.visualize_hierarchy.assert_called_once_with(
            hierarchical_topics=hierarchical_topics
        )
        figure.update_layout.assert_called_once_with(width=None, autosize=True)
        figure.to_html.assert_called_once_with(
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True},
        )

    @patch("bruno.app._load_bertopic_model")
    def test_hierarchy_visualization_requires_two_topics(self, load_topic_model):
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: []}),
            visualize_hierarchy=Mock(),
        )
        load_topic_model.return_value = topic_model
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()

        response = self.client.get("/visualizations/hierarchy")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"requires at least two topics", response.data)
        topic_model.visualize_hierarchy.assert_not_called()

    @patch("bruno.app._load_bertopic_model")
    def test_heatmap_visualization_renders_saved_model(self, load_topic_model):
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Rendered heatmap</div>"),
        )
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            visualize_heatmap=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()

        response = self.client.get("/visualizations/heatmap")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rendered heatmap", response.data)
        topic_model.visualize_heatmap.assert_called_once_with()
        figure.update_layout.assert_called_once_with(width=None, autosize=True)
        figure.to_html.assert_called_once_with(
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True},
        )

    @patch("bruno.app._load_bertopic_model")
    def test_heatmap_visualization_requires_two_topics(self, load_topic_model):
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: []}),
            visualize_heatmap=Mock(),
        )
        load_topic_model.return_value = topic_model
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()

        response = self.client.get("/visualizations/heatmap")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"requires at least two topics", response.data)
        topic_model.visualize_heatmap.assert_not_called()

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_topics_over_time_uses_each_thinkers_approximate_year(
        self, split_documents, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        _save_topic_source_documents(
            model_path,
            ["First full text", "Second full text", "Third full text"],
            [
                "First Thinker (1700-1760) (active 1730-1750)",
                "Second Thinker (400 BC-300 BC)",
                "Third Thinker (1840-1880) (active 1845-1865)",
            ],
        )
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Topics over time</div>"),
        )
        time_topics = object()
        topic_model = SimpleNamespace(
            topics_over_time=Mock(return_value=time_topics),
            visualize_topics_over_time=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        split_documents.return_value = (
            [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ],
            ["First Thinker", "First Thinker", "Second Thinker", "Third Thinker"],
            [0, 0, 1, 2],
        )

        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module
        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/topics-over-time")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Topics over time", response.data)
        self.assertIn(
            b"For readability, all approximate years before 1800 are displayed as 1800.",
            response.data,
        )
        self.assertEqual(
            split_documents.call_args.args[:3],
            (
                ["First full text", "Second full text", "Third full text"],
                [
                    "First Thinker (1700-1760) (active 1730-1750)",
                    "Second Thinker (400 BC-300 BC)",
                    "Third Thinker (1840-1880) (active 1845-1865)",
                ],
                [0, 1, 2],
            ),
        )
        topic_model.topics_over_time.assert_called_once_with(
            [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
                "Fourth sentence.",
            ],
            [1800, 1800, 1800, 1855],
        )
        topic_model.visualize_topics_over_time.assert_called_once_with(
            time_topics
        )
        figure.update_layout.assert_called_once_with(width=None, autosize=True)

    @patch("bruno.app._load_bertopic_model")
    def test_topics_over_time_requires_a_year_for_every_thinker(
        self, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        _save_topic_source_documents(
            model_path,
            ["Source text"],
            ["Undated Thinker"],
        )
        load_topic_model.return_value = SimpleNamespace(
            topics_over_time=Mock(),
            visualize_topics_over_time=Mock(),
        )

        response = self.client.get("/visualizations/topics-over-time")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"requires an approximate year for every thinker", response.data)
        load_topic_model.return_value.topics_over_time.assert_not_called()

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_topics_per_class_uses_all_thinkers_as_classes(
        self, split_documents, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        thinker_names = [
            "First Thinker (1700-1760)",
            "Second Thinker (1900-1980)",
        ]
        _save_topic_source_documents(
            model_path,
            ["First source text", "Second source text"],
            thinker_names,
        )
        figure = SimpleNamespace(
            update_layout=Mock(),
            update_yaxes=Mock(),
            to_html=Mock(return_value="<div>Topics per thinker</div>"),
        )
        topics_per_class = object()
        topic_model = SimpleNamespace(
            topics_per_class=Mock(return_value=topics_per_class),
            visualize_topics_per_class=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        split_documents.return_value = (
            ["First sentence.", "Second sentence.", "Third sentence."],
            [thinker_names[0], thinker_names[0], thinker_names[1]],
            [0, 0, 1],
        )

        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module
        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/topics-per-class")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Topics per thinker", response.data)
        self.assertEqual(
            split_documents.call_args.args[:3],
            (
                ["First source text", "Second source text"],
                thinker_names,
                [0, 1],
            ),
        )
        topic_model.topics_per_class.assert_called_once_with(
            ["First sentence.", "Second sentence.", "Third sentence."],
            ["First Thinker", "First Thinker", "Second Thinker"],
        )
        topic_model.visualize_topics_per_class.assert_called_once_with(
            topics_per_class
        )
        figure.update_yaxes.assert_called_once_with(
            categoryorder="array",
            categoryarray=["First Thinker", "Second Thinker"],
            autorange="reversed",
        )
        figure.update_layout.assert_called_once_with(width=None, autosize=True)

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_visualize_documents_hides_document_hover(
        self, split_documents, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        _save_topic_source_documents(
            model_path, ["Source text"], ["First Thinker (1700-1760)"]
        )
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Documents</div>"),
        )
        topic_model = SimpleNamespace(
            visualize_documents=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        split_documents.return_value = (
            ["First sentence."],
            ["First Thinker (1700-1760)"],
            [0],
        )
        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module
        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/visualize-documents")

        self.assertEqual(response.status_code, 200)
        topic_model.visualize_documents.assert_called_once_with(
            ["First sentence."],
            hide_document_hover=True,
        )

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_hierarchical_documents_visualization_uses_hierarchical_topics(
        self, split_documents, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        _save_topic_source_documents(
            model_path, ["Source text"], ["First Thinker (1700-1760)"]
        )
        figure = SimpleNamespace(
            update_layout=Mock(),
            to_html=Mock(return_value="<div>Hierarchical documents</div>"),
        )
        hierarchical_topics = [object()]
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={-1: [], 0: [], 1: []}),
            hierarchical_topics=Mock(return_value=hierarchical_topics),
            visualize_hierarchical_documents=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        split_documents.return_value = (
            ["First sentence.", "Second sentence."],
            ["First Thinker (1700-1760)"] * 2,
            [0, 0],
        )
        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module
        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/hierarchical-documents")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Hierarchical documents", response.data)
        topic_model.hierarchical_topics.assert_called_once_with(
            ["First sentence.", "Second sentence."]
        )
        topic_model.visualize_hierarchical_documents.assert_called_once_with(
            ["First sentence.", "Second sentence."],
            hierarchical_topics,
            nr_levels=1,
            hide_document_hover=True,
        )

    @patch("bruno.app._load_bertopic_model")
    @patch("bruno.app._split_documents_into_sentences")
    def test_topics_per_class_uses_all_thinkers_as_classes(
        self, split_documents, load_topic_model
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        thinker_names = [
            "First Thinker (1700-1760)",
            "Second Thinker (1900-1980)",
        ]
        _save_topic_source_documents(
            model_path,
            ["First source text", "Second source text"],
            thinker_names,
        )
        figure = SimpleNamespace(
            update_layout=Mock(),
            update_yaxes=Mock(),
            to_html=Mock(return_value="<div>Topics per thinker</div>"),
        )
        topics_per_class = object()
        topic_model = SimpleNamespace(
            topics_per_class=Mock(return_value=topics_per_class),
            visualize_topics_per_class=Mock(return_value=figure),
        )
        load_topic_model.return_value = topic_model
        split_documents.return_value = (
            [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ],
            [thinker_names[0], thinker_names[0], thinker_names[1]],
            [0, 0, 1],
        )

        tokenize_module = ModuleType("nltk.tokenize")
        tokenize_module.sent_tokenize = Mock()
        nltk_module = ModuleType("nltk")
        nltk_module.tokenize = tokenize_module
        with patch.dict(
            "sys.modules",
            {"nltk": nltk_module, "nltk.tokenize": tokenize_module},
        ):
            response = self.client.get("/visualizations/topics-per-class")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Topics per thinker", response.data)
        self.assertEqual(
            split_documents.call_args.args[:3],
            (
                ["First source text", "Second source text"],
                thinker_names,
                [0, 1],
            ),
        )
        topic_model.topics_per_class.assert_called_once_with(
            [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ],
            ["First Thinker", "First Thinker", "Second Thinker"],
        )
        topic_model.visualize_topics_per_class.assert_called_once_with(
            topics_per_class
        )
        figure.update_yaxes.assert_called_once_with(
            categoryorder="array",
            categoryarray=["First Thinker", "Second Thinker"],
            autorange="reversed",
        )
        figure.update_layout.assert_called_once_with(width=None, autosize=True)

    def test_topic_source_documents_are_saved_with_their_thinker_names(self):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()

        _save_topic_source_documents(
            model_path,
            ["First source text", "Second source text"],
            ["First thinker", "Second thinker"],
        )

        self.assertEqual(
            _load_topic_source_documents(
                model_path / "topic_source_documents.json"
            ),
            [
                {"name": "First thinker", "text": "First source text"},
                {"name": "Second thinker", "text": "Second source text"},
            ],
        )

    @patch("bruno.app._load_bertopic_model")
    def test_approximate_distribution_visualization_uses_saved_source_documents(
        self, load_topic_model
    ):
        first_table = Mock()
        first_table.format.return_value = first_table
        first_table.to_html.return_value = "<table>First distribution</table>"
        second_table = Mock()
        second_table.format.return_value = second_table
        second_table.to_html.return_value = "<table>Second distribution</table>"
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            approximate_distribution=Mock(
                return_value=(
                    [[0.2, 0.8], [0.7, 0.3]],
                    ["first tokens", "second tokens"],
                )
            ),
            visualize_approximate_distribution=Mock(
                side_effect=[first_table, second_table]
            ),
        )
        load_topic_model.return_value = topic_model
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        (model_path / "topic_source_documents.json").write_text(
            json.dumps(
                [
                    {"name": "<First thinker>", "text": "First source text"},
                    {"name": "Second thinker", "text": "Second source text"},
                ]
            ),
            encoding="utf-8",
        )
        (model_path / "topic_search_vectorizer.json").write_text(
            "{}",
            encoding="utf-8",
        )

        response = self.client.get(
            "/visualizations/approximate-distribution"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<h2>&lt;First thinker&gt;</h2>", response.data)
        self.assertIn(
            b'data-search-text="&lt;first thinker&gt;"',
            response.data,
        )
        self.assertIn(b"<table>First distribution</table>", response.data)
        self.assertIn(b"<table>Second distribution</table>", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        load_topic_model.assert_called_once_with(model_path)
        self.assertEqual(
            [
                call.args
                for call in topic_model.approximate_distribution.call_args_list
            ],
            [(["First source text", "Second source text"],)],
        )
        self.assertEqual(
            [
                call.kwargs
                for call in topic_model.approximate_distribution.call_args_list
            ],
            [{"use_embedding_model": False, "calculate_tokens": True}],
        )
        self.assertEqual(
            [
                call.args
                for call in topic_model.visualize_approximate_distribution.call_args_list
            ],
            [
                ("First source text", "first tokens"),
                ("Second source text", "second tokens"),
            ],
        )
        first_table.format.assert_called_once_with(escape="html")
        second_table.format.assert_called_once_with(escape="html")
        first_table.to_html.assert_called_once_with()
        second_table.to_html.assert_called_once_with()

    @patch("bruno.app._load_bertopic_model")
    def test_approximate_distributions_api_returns_topic_scores_by_thinker(
        self, load_topic_model
    ):
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            get_topic=Mock(
                side_effect=[
                    [("maneuver", 0.5)],
                    [("logistics", 0.5)],
                ]
            ),
            approximate_distribution=Mock(
                return_value=(
                    [[0.2, 0.8], [0.7, 0.3]],
                    ["first tokens", "second tokens"],
                )
            ),
        )
        load_topic_model.return_value = topic_model
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        (model_path / "topic_source_documents.json").write_text(
            json.dumps(
                [
                    {"name": "First thinker", "text": "First source text"},
                    {"name": "Second thinker", "text": "Second source text"},
                ]
            ),
            encoding="utf-8",
        )
        (model_path / "topic_search_vectorizer.json").write_text(
            "{}",
            encoding="utf-8",
        )

        connection = sqlite3.connect(self.app.config["DATABASE"])
        try:
            connection.execute(
                "INSERT INTO thinkers (name, military_thinking_document_id) "
                "VALUES ('First thinker', 'doc-first')"
            )
            connection.commit()
        finally:
            connection.close()

        response = self.client.get(
            "/api/visualizations/approximate-distributions"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {
                "topics": [
                    {
                        "topic_id": 0,
                        "name": "maneuver",
                        "thinkers": [
                            {
                                "name": "First thinker",
                                "document_id": "doc-first",
                                "probability": 0.2,
                            },
                            {
                                "name": "Second thinker",
                                "document_id": None,
                                "probability": 0.7,
                            },
                        ],
                    },
                    {
                        "topic_id": 1,
                        "name": "logistics",
                        "thinkers": [
                            {
                                "name": "First thinker",
                                "document_id": "doc-first",
                                "probability": 0.8,
                            },
                            {
                                "name": "Second thinker",
                                "document_id": None,
                                "probability": 0.3,
                            },
                        ],
                    },
                ]
            },
        )
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        topic_model.approximate_distribution.assert_called_once_with(
            ["First source text", "Second source text"],
            use_embedding_model=False,
            calculate_tokens=True,
        )

    @patch("bruno.app._load_bertopic_model")
    def test_approximate_distribution_requires_saved_source_documents(
        self, load_topic_model
    ):
        load_topic_model.return_value = SimpleNamespace(
            visualize_approximate_distribution=Mock()
        )
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()

        response = self.client.get(
            "/visualizations/approximate-distribution"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Regenerate topics to save their source texts",
            response.data,
        )
        load_topic_model.return_value.visualize_approximate_distribution.assert_not_called()

    @patch("bruno.app._load_bertopic_model")
    def test_approximate_distribution_requires_saved_embedding_vectorizer(
        self, load_topic_model
    ):
        load_topic_model.return_value = SimpleNamespace()
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        (model_path / "topic_source_documents.json").write_text(
            json.dumps([{"name": "Thinker", "text": "Source text"}]),
            encoding="utf-8",
        )

        response = self.client.get(
            "/visualizations/approximate-distribution"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Regenerate topics to save their text embedding vectorizer",
            response.data,
        )

    def test_find_topics_requires_a_saved_model(self):
        response = self.client.post(
            "/api/concepts/find-topics",
            json={"concept": "logistics"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Generate topics", response.json["error"])

    def test_find_topics_rejects_invalid_search_payloads(self):
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()
        cases = [
            ({"concept": ""}, "search text must be"),
            ({"concept": "  "}, "search text must be"),
            ({"concept": "x" * 501}, "at most 500"),
            (
                {"concept": "logistics", "top_n": 5},
                "Unknown field",
            ),
        ]

        for payload, expected_error in cases:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/concepts/find-topics",
                    json=payload,
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected_error, response.json["error"])

    @patch("bruno.app._load_topic_search_embedder")
    @patch("bruno.app._load_bertopic_model")
    def test_find_topics_loads_model_and_returns_similar_topics(
        self,
        load_topic_model,
        load_query_embedder,
    ):
        model_path = Path(self.app.config["BERTOPIC_MODEL_PATH"])
        model_path.mkdir()
        (model_path / "topic_search_vectorizer.json").write_text(
            "{}",
            encoding="utf-8",
        )
        self.client.post("/api/concepts", json={"name": "First concept"})
        self.client.post("/api/concepts", json={"name": "Second concept"})
        query_embedder = object()
        load_query_embedder.return_value = query_embedder
        topic_model = SimpleNamespace(
            topic_embeddings_=SimpleNamespace(ndim=2),
            get_topics=Mock(return_value={0: [], 1: []}),
            find_topics=Mock(return_value=([1, 0], [0.82, 0.33])),
            get_topic=Mock(
                side_effect={
                    0: [("strategy", 0.8), ("war", 0.5)],
                    1: [("logistics", 0.7), ("supply", 0.4)],
                }.get
            ),
        )
        load_topic_model.return_value = topic_model

        response = self.client.post(
            "/api/concepts/find-topics",
            json={"concept": " military logistics "},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(
            response.json["results"],
            [
                {
                    "topic_id": 1,
                    "name": "logistics · supply",
                    "similarity": 0.82,
                },
                {
                    "topic_id": 0,
                    "name": "strategy · war",
                    "similarity": 0.33,
                },
            ],
        )
        load_topic_model.assert_called_once_with(model_path)
        load_query_embedder.assert_called_once_with(
            model_path / "topic_search_vectorizer.json"
        )
        self.assertIs(topic_model.embedding_model, query_embedder)
        topic_model.find_topics.assert_called_once_with(
            search_term="military logistics",
            top_n=2,
        )

    def test_find_topics_requires_search_vectorizer_for_saved_models(self):
        Path(self.app.config["BERTOPIC_MODEL_PATH"]).mkdir()
        self.client.post("/api/concepts", json={"name": "Existing concept"})

        response = self.client.post(
            "/api/concepts/find-topics",
            json={"concept": "logistics"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Regenerate topics", response.json["error"])

    def test_topic_search_vectorizer_saves_integer_indices_as_json(self):
        class IntegerIndex:
            def __int__(self):
                return 0

        model_path = Path(self.temp_dir.name) / "saved-model"
        model_path.mkdir()
        vectorizer = SimpleNamespace(
            vocabulary_={"logistics": IntegerIndex()},
            idf_=SimpleNamespace(tolist=lambda: [1.25]),
            get_params=Mock(
                return_value={
                    "ngram_range": (1, 5),
                    "stop_words": "english",
                }
            ),
        )

        _save_topic_search_vectorizer(model_path, vectorizer)

        saved_vectorizer = json.loads(
            (model_path / "topic_search_vectorizer.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(saved_vectorizer["vocabulary"], {"logistics": 0})
        self.assertEqual(saved_vectorizer["idf"], [1.25])
        self.assertEqual(saved_vectorizer["ngram_range"], [1, 5])
        self.assertEqual(saved_vectorizer["stop_words"], "english")

    def test_load_bertopic_model_restores_tfidf_embedding_space(self):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        model_path = Path(self.temp_dir.name) / "saved-model"
        model_path.mkdir()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
        )
        vectorizer.fit(
            ["military strategy and logistics", "operational planning and maneuver"]
        )
        _save_topic_search_vectorizer(model_path, vectorizer)
        load_model = Mock(return_value=SimpleNamespace())
        bertopic_module = ModuleType("bertopic")
        backend_module = ModuleType("bertopic.backend")

        class FakeBaseEmbedder:
            def __init__(self):
                pass

        backend_module.BaseEmbedder = FakeBaseEmbedder
        bertopic_module.BERTopic = SimpleNamespace(load=load_model)
        texts = ["military strategy planning"]

        with patch.dict(
            "sys.modules",
            {
                "bertopic": bertopic_module,
                "bertopic.backend": backend_module,
            },
        ):
            _load_bertopic_model(model_path)

        embedder = load_model.call_args.kwargs["embedding_model"]
        self.assertIsInstance(embedder, FakeBaseEmbedder)
        np.testing.assert_allclose(
            embedder.embed(texts).toarray(),
            vectorizer.transform(texts).toarray(),
        )

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_rejects_unknown_topic_source(self, generate_topics):
        self.add_thinker_text("Clausewitz", "Clausewitz source text")

        for topic_source in ("other", [], {}):
            with self.subTest(topic_source=topic_source):
                response = self.client.post(
                    "/api/concepts/generate",
                    json={"topic_source": topic_source},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "topic_source must be 'ollama' or 'wikipedia'",
                    response.json["error"],
                )

        generate_topics.assert_not_called()

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_requires_saved_ollama_analysis_when_selected(
        self, generate_topics
    ):
        self.add_thinker_text("Clausewitz", "Wikipedia text only for this test")
        thinker = self.client.get("/api/thinkers").json[0]
        ollama_id = thinker["military_thinking_document_id"]
        self.mongo_client.documents.delete_many(
            {"_id": {"$in": [ObjectId(ollama_id)]}}
        )

        response = self.client.post(
            "/api/concepts/generate", json={"topic_source": "ollama"}
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Ollama analysis", response.json["error"])
        self.assertIn("Clausewitz", response.json["error"])
        self.download_wikipedia_text.assert_not_called()
        generate_topics.assert_not_called()

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_rejects_topic_count_below_one_or_non_integer(
        self, generate_topics
    ):
        for topic_count in (0, -1, True, 1.5, "2"):
            with self.subTest(topic_count=topic_count):
                response = self.client.post(
                    "/api/concepts/generate",
                    json={"topic_count": topic_count},
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("integer of at least 1", response.json["error"])
        generate_topics.assert_not_called()

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_rejects_more_topics_than_available_texts(
        self, generate_topics
    ):
        self.add_thinker_text("Clausewitz", "PLAINTEXT clausewitz")

        response = self.client.post(
            "/api/concepts/generate", json={"topic_count": 2}
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("cannot exceed the number of thinker texts", response.json["error"])
        generate_topics.assert_not_called()

    def test_concepts_use_approximate_distributions_for_each_topic(self):
        topic_terms = {
            0: [
                ("strategy", 0.8),
                ("war", 0.5),
                ("tactics", 0.4),
                ("politics", 0.3),
                ("doctrine", 0.2),
            ],
            1: [],
            2: [("logistics", 0.5)],
        }
        distributions = [
            {"document": {"name": name}, "topic_ids": [0, 1, 2],
             "probabilities": probabilities}
            for name, probabilities in [
                ("Clausewitz", [0.8, 0.1, 0.1]),
                ("Sun Tzu", [0.65, 0.2, 0.15]),
                ("Jomini", [0.1, 0.6, 0.3]),
            ]
        ]

        concepts = _concepts_from_distributions(
            distributions, [1, 2, 3], topic_terms.get
        )

        self.assertEqual(
            [concept["name"] for concept in concepts],
            [
                "strategy · war · tactics · politics · doctrine",
                "Topic 1",
                "logistics",
            ],
        )
        self.assertEqual(concepts[0]["thinker_ids"], [1, 2])
        self.assertEqual(concepts[1]["thinker_ids"], [3])
        # Topic 2 is nobody's strongest, so its best-matching thinker is listed.
        self.assertEqual(concepts[2]["thinker_ids"], [3])
        self.assertEqual(
            concepts[0]["topic_assignments"],
            [
                {"thinker_id": 1, "topic_id": 0, "probability": 0.8},
                {"thinker_id": 2, "topic_id": 0, "probability": 0.65},
            ],
        )
        self.assertEqual(
            concepts[2]["topic_assignments"],
            [{"thinker_id": 3, "topic_id": 2, "probability": 0.3}],
        )

    def test_approximate_distributions_are_calculated_once_and_cached(self):
        model_path = Path(self.temp_dir.name) / "model"
        model_path.mkdir()
        topic_model = SimpleNamespace(
            get_topics=Mock(return_value={0: [], 1: []}),
            approximate_distribution=Mock(
                return_value=([[0.2, 0.8], [0.7, 0.3]], ["a", "b"])
            ),
        )
        documents = [
            {"name": "First", "text": "First text"},
            {"name": "Second", "text": "Second text"},
        ]

        first = _approximate_distributions(topic_model, documents, model_path)
        second = _approximate_distributions(topic_model, documents, model_path)

        topic_model.approximate_distribution.assert_called_once()
        self.assertEqual(
            [item["probabilities"] for item in second],
            [[0.2, 0.8], [0.7, 0.3]],
        )
        self.assertEqual(
            [item["document"]["name"] for item in first],
            [item["document"]["name"] for item in second],
        )
    def test_sentence_splitting_preserves_thinker_mapping(self):
        documents, names, thinker_ids = _split_documents_into_sentences(
            ["First.|Second!", "Another sentence."],
            ["Clausewitz", "Sun Tzu"],
            [11, 12],
            lambda document: document.split("|"),
        )

        self.assertEqual(
            documents,
            ["First.", "Second!", "Another sentence."],
        )
        self.assertEqual(names, ["Clausewitz", "Clausewitz", "Sun Tzu"])
        self.assertEqual(thinker_ids, [11, 11, 12])

    def test_legacy_generated_concepts_migrate_thinker_links(self):
        thinker_id = self.add_thinker_text(
            "Clausewitz", "Clausewitz text"
        )
        self.client.post(
            "/api/concepts",
            json={
                "name": "war · strategy",
                "description": (
                    "BERTopic found this theme in the Wikipedia text of "
                    "Clausewitz. Key terms: war, strategy, tactics, politics, doctrine."
                ),
            },
        )

        restarted_app = create_app(
            Path(self.temp_dir.name) / "test.sqlite3",
            mongo_client=self.mongo_client,
        )
        migrated_concept = restarted_app.test_client().get(
            "/api/concepts"
        ).json[0]

        self.assertEqual(
            migrated_concept["name"],
            "war · strategy · tactics · politics · doctrine",
        )
        self.assertEqual(migrated_concept["description"], "")
        self.assertEqual(
            migrated_concept["thinkers"],
            [
                {
                    "id": thinker_id,
                    "name": "Clausewitz",
                    "wiki_text_document_id": self.client.get(
                        "/api/thinkers"
                    ).json[0]["wiki_text_document_id"],
                    "military_thinking_document_id": self.client.get(
                        "/api/thinkers"
                    ).json[0]["military_thinking_document_id"],
                }
            ],
        )

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_rejects_thinkers_without_saved_plain_text(
        self, generate_topics
    ):
        connection = sqlite3.connect(self.app.config["DATABASE"])
        try:
            connection.execute(
                "INSERT INTO thinkers (name) VALUES (?)",
                ("Missing text",),
            )
            connection.commit()
        finally:
            connection.close()

        response = self.client.post(
            "/api/concepts/generate", json={"topic_source": "wikipedia"}
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Missing text", response.json["error"])
        generate_topics.assert_not_called()
        self.assertEqual(self.client.get("/api/concepts").json, [])

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_fetches_and_saves_missing_extracted_wikipedia_text(
        self, generate_topics
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Legacy_Thinker"
        connection = sqlite3.connect(self.app.config["DATABASE"])
        try:
            cursor = connection.execute(
                "INSERT INTO thinkers (name, biography, wiki_document_key) "
                "VALUES (?, ?, ?)",
                ("Legacy Thinker", wikipedia_url, "legacy-thinker-docs"),
            )
            connection.commit()
            thinker_id = cursor.lastrowid
        finally:
            connection.close()
        generate_topics.return_value = [
            {"name": "military · strategy", "thinker_ids": [thinker_id]}
        ]

        response = self.client.post(
            "/api/concepts/generate", json={"topic_source": "wikipedia"}
        )

        self.assertEqual(response.status_code, 201)
        self.download_wikipedia_text.assert_called_once_with(wikipedia_url)
        self.assertEqual(
            generate_topics.call_args.args,
            (
                ["Clausewitz developed influential ideas on war."],
                ["Legacy Thinker"],
                None,
                [thinker_id],
            ),
        )
        saved_thinker = self.client.get("/api/thinkers").json[0]
        self.assertIsNotNone(saved_thinker["wiki_text_document_id"])
        saved_source = self.client.get(
            f"/api/documents/{saved_thinker['wiki_text_document_id']}"
        )
        self.assertEqual(
            saved_source.get_data(as_text=True),
            "Clausewitz developed influential ideas on war.",
        )
        saved_source.close()

    @patch("bruno.app._generate_bertopic_concepts")
    def test_generate_concepts_requires_thinkers_and_empty_request_body(
        self, generate_topics
    ):
        no_thinkers = self.client.post("/api/concepts/generate", json={})
        client_texts = self.client.post(
            "/api/concepts/generate", json={"texts": ["not from a thinker"]}
        )

        self.assertEqual(no_thinkers.status_code, 409)
        self.assertEqual(client_texts.status_code, 400)
        generate_topics.assert_not_called()

    @patch("bruno.app._generate_bertopic_concepts", side_effect=ValueError("empty vocabulary"))
    def test_generate_concepts_reports_topic_model_errors_without_saving(
        self, generate_topics
    ):
        self.add_thinker_text("Clausewitz", "text")
        self.client.post("/api/concepts", json={"name": "Existing concept"})

        response = self.client.post("/api/concepts/generate", json={})

        self.assertEqual(response.status_code, 422)
        self.assertIn("BERTopic could not generate topics", response.json["error"])
        self.assertEqual(self.client.get("/api/concepts").json, [])
        generate_topics.assert_called_once()

    @patch("bruno.app.urlopen")
    def test_extract_military_contributions_sends_wikipedia_text_to_model(
        self, open_url
    ):
        expected = "He emphasized the moral and political dimensions of war."
        open_url.return_value = io.BytesIO(
            json.dumps({"response": expected}).encode()
        )

        contributions = _extract_military_contributions(
            "Clausewitz on war",
            word_count=200,
        )

        self.assertEqual(contributions, expected)
        model_request = open_url.call_args.args[0]
        request_payload = json.loads(model_request.data)
        self.assertEqual(
            request_payload["prompt"].splitlines()[0],
            "What are the thinker's contributions to military strategic thought?",
        )
        self.assertIn(
            "Give a concise answer, approximately 200 words",
            request_payload["prompt"],
        )
        self.assertIn(
            "Do not include named entities",
            request_payload["prompt"],
        )
        self.assertIn(
            "omit names of people, places, organizations, battles, and countries",
            request_payload["prompt"],
        )
        self.assertIn(
            "Describe strategic ideas in general terms",
            request_payload["prompt"],
        )
        self.assertIn(
            "What are the thinker's contributions to military strategic thought?",
            request_payload["prompt"],
        )
        self.assertLess(
            request_payload["prompt"].index(
                "Describe strategic ideas in general terms."
            ),
            request_payload["prompt"].index(
                "Make use of the provided Wikipedia article and your foundational knowledge on this thinker."
            ),
        )
        self.assertNotIn("in bullet points", request_payload["prompt"])
        self.assertIn("Clausewitz on war", request_payload["prompt"])
        self.assertIn("Wikipedia article text:", request_payload["prompt"])
        self.assertIn("Do not mention the thinker's name", request_payload["prompt"])
        self.assertIn("years and specific dates", request_payload["prompt"])
        self.assertEqual(
            request_payload["options"],
            {"num_ctx": 8192, "num_predict": 300},
        )
        self.assertEqual(request_payload["keep_alive"], "10m")
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 900})

    @patch("bruno.app.urlopen")
    def test_extract_military_contributions_uses_requested_word_count(
        self, open_url
    ):
        open_url.return_value = io.BytesIO(
            json.dumps({"response": "Requested-length response."}).encode()
        )

        _extract_military_contributions("Clausewitz on war", word_count=400)

        request_payload = json.loads(open_url.call_args.args[0].data)
        self.assertIn(
            "Give a concise answer, approximately 400 words",
            request_payload["prompt"],
        )
        self.assertEqual(
            request_payload["options"],
            {"num_ctx": 8192, "num_predict": 600},
        )

    @patch("bruno.app._download_remote_content")
    def test_download_wikipedia_text_uses_plaintext_mediawiki_api(self, download):
        download.return_value = (
            json.dumps(
                {
                    "query": {
                        "pages": {
                            "123": {
                                "extract": "  Article text\nwith plain text.  "
                            }
                        }
                    }
                }
            ).encode(),
            "https://en.wikipedia.org/w/api.php",
        )

        text = _download_wikipedia_text(
            "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        )

        self.assertEqual(text, "Article text\nwith plain text.")
        request_url = download.call_args.args[0]
        self.assertEqual(urlsplit(request_url).path, "/w/api.php")
        self.assertEqual(
            parse_qs(urlsplit(request_url).query),
            {
                "action": ["query"],
                "format": ["json"],
                "prop": ["extracts"],
                "explaintext": ["1"],
                "redirects": ["1"],
                "titles": ["Carl_von_Clausewitz"],
            },
        )
        self.assertEqual(
            download.call_args.kwargs,
            {
                "allowed_hosts": ("wikipedia.org",),
                "expected_content_type": "application/json",
            },
        )

    
    @patch("bruno.app._download_remote_content")
    def test_download_dbpedia_data_uses_longer_timeout(self, download):
        download.return_value = (b"{}", "https://dbpedia.org/data/Article.json")

        content, final_url = _download_dbpedia_data(
            "https://dbpedia.org/data/Article.json"
        )

        self.assertEqual(content, b"{}")
        self.assertEqual(final_url, "https://dbpedia.org/data/Article.json")
        self.assertEqual(
            download.call_args.kwargs,
            {
                "allowed_hosts": ("dbpedia.org",),
                "expected_content_type": "application/json",
                "timeout": 120,
            },
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

        restarted_app = create_app(
            Path(self.temp_dir.name) / "test.sqlite3",
            mongo_client=self.mongo_client,
        )
        restarted_client = restarted_app.test_client()

        self.assertEqual(restarted_client.get("/api/concepts").json[0]["name"], "Plurality")

    def test_missing_records_return_json_404(self):
        response = self.client.get("/api/thinkers/999")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json, {"error": "Thinker not found."})

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_downloads_and_saves_both_sources(
        self, download_wikipedia, download_dbpedia
    ):
        page_content = b"<!doctype html><title>Clausewitz</title>"
        wikipedia_url = "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        download_wikipedia.return_value = (
            page_content,
            wikipedia_url,
        )
        dbpedia_content = dbpedia_person(
            "Carl_von_Clausewitz", "Carl von Clausewitz"
        )
        dbpedia_url = "https://dbpedia.org/data/Carl_von_Clausewitz.json"
        download_dbpedia.return_value = (
            dbpedia_content,
            dbpedia_url,
        )
        
        response = self.client.post(
            "/api/thinkers/process",
            json={
                "biography": wikipedia_url,
                "word_count": 100,
                "prompt": "Summarize his strategic principles.",
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json["name"],
            "Carl von Clausewitz (1780-1831) (active 1792-1831)",
        )
        self.assertEqual(len(self.mongo_client.documents.records), 4)
        self.assertEqual(response.json["wiki_document_id"], next(
            key for key, document in self.mongo_client.documents.records.items()
            if document["source"] == "wikipedia"
        ))
        self.assertEqual(response.json["wiki_text_document_id"], next(
            key for key, document in self.mongo_client.documents.records.items()
            if document["source"] == "wikipedia-text"
        ))
        self.assertEqual(response.json["dbpedia_document_id"], next(
            key for key, document in self.mongo_client.documents.records.items()
            if document["source"] == "dbpedia"
        ))
        self.assertEqual(response.json["military_thinking_document_id"], next(
            key for key, document in self.mongo_client.documents.records.items()
            if document["source"] == "military-strategy"
        ))
        self.download_wikipedia_text.assert_called_once_with(wikipedia_url)
        saved_text = self.client.get(
            f"/api/documents/{response.json['wiki_text_document_id']}"
        )
        self.assertEqual(saved_text.status_code, 200)
        self.assertEqual(
            saved_text.get_data(as_text=True),
            "Clausewitz developed influential ideas on war.",
        )
        self.assertEqual(saved_text.mimetype, "text/plain")
        text_view = self.client.get(
            f"/documents/{response.json['wiki_text_document_id']}"
        )
        self.assertEqual(text_view.status_code, 200)
        self.assertIn(b'class="document-page"', text_view.data)
        self.assertIn(b'<p class="eyebrow">Document</p>', text_view.data)
        self.assertIn(b"Clausewitz developed influential ideas on war.", text_view.data)
        text_view.close()
        saved_text.close()
        saved_contributions = self.client.get(
            f"/api/documents/{response.json['military_thinking_document_id']}"
        )
        self.assertEqual(saved_contributions.status_code, 200)
        self.assertEqual(
            saved_contributions.get_data(as_text=True),
            "Clausewitz developed influential ideas on military strategy.",
        )
        self.assertEqual(saved_contributions.mimetype, "text/plain")
        contributions_view = self.client.get(
            f"/documents/{response.json['military_thinking_document_id']}"
        )
        self.assertEqual(contributions_view.status_code, 200)
        self.assertIn(b'<p class="eyebrow">Document</p>', contributions_view.data)
        self.assertIn(b"Ollama analysis", contributions_view.data)
        self.assertIn(b"What is Ollama?", contributions_view.data)
        self.assertIn(b"running large language models locally", contributions_view.data)
        self.assertIn(b"Prompt sent to Ollama", contributions_view.data)
        self.assertEqual(contributions_view.data.count(b"<details>"), 2)
        self.assertNotIn(b"<details open>", contributions_view.data)
        self.assertEqual(contributions_view.data.count(b"<blockquote>"), 1)
        self.assertIn(
            b"Summarize his strategic principles.",
            contributions_view.data,
        )
        self.assertIn(
            b"Do not include named entities",
            contributions_view.data,
        )
        self.assertIn(
            b"Summarize his strategic principles.",
            contributions_view.data,
        )
        self.assertIn(b"approximately 100 words", contributions_view.data)
        self.assertIn(
            b"Wikipedia article text:\nClausewitz developed influential ideas on war.",
            contributions_view.data,
        )
        self.assertIn(
            b"Clausewitz developed influential ideas on military strategy.",
            contributions_view.data,
        )
        contributions_view.close()
        saved_contributions.close()
        served_page = self.client.get(
            f"/api/documents/{response.json['wiki_document_id']}"
        )
        self.assertEqual(served_page.status_code, 200)
        self.assertEqual(served_page.get_data(), page_content)
        served_page.close()
        served_dbpedia = self.client.get(
            f"/api/documents/{response.json['dbpedia_document_id']}"
        )
        self.assertEqual(served_dbpedia.status_code, 200)
        self.assertEqual(served_dbpedia.get_data(), dbpedia_content)
        self.assertEqual(served_dbpedia.mimetype, "application/json")
        served_dbpedia.close()
        dbpedia_view = self.client.get(
            f"/documents/{response.json['dbpedia_document_id']}"
        )
        self.assertEqual(dbpedia_view.status_code, 200)
        self.assertIn(b"<h1 class=\"document-title\">DBpedia data</h1>", dbpedia_view.data)
        self.assertIn(b"What is DBpedia?", dbpedia_view.data)
        self.assertIn(b"linked open data", dbpedia_view.data)
        self.assertNotIn(b"Taxonomy", dbpedia_view.data)
        self.assertIn(b"DBpedia classes", dbpedia_view.data)
        self.assertIn(b">Person</a>", dbpedia_view.data)
        self.assertIn(b"Wikipedia categories", dbpedia_view.data)
        self.assertIn(b">Military theorists</a>", dbpedia_view.data)
        self.assertNotIn(b"application/json", dbpedia_view.data)
        dbpedia_view.close()
        self.assertEqual(
            self.client.get("/api/thinkers").json[0]["wiki_document_id"],
            response.json["wiki_document_id"],
        )
        self.assertFalse((Path(self.temp_dir.name) / "wiki").exists())
        self.assertFalse((Path(self.temp_dir.name) / "dbpedia").exists())
        download_wikipedia.assert_called_once_with(wikipedia_url)
        download_dbpedia.assert_called_once_with(dbpedia_url)

    def test_process_thinker_rejects_invalid_response_word_counts(self):
        for word_count in (0, 1, 99, 1001, True, 1.5, "200"):
            with self.subTest(word_count=word_count):
                response = self.client.post(
                    "/api/thinkers/process",
                    json={
                        "biography": "https://en.wikipedia.org/wiki/Carl_von_Clausewitz",
                        "word_count": word_count,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn("integer between 100 and 1000", response.json["error"])
        self.extract_military_contributions.assert_not_called()
        self.extract_military_contributions.assert_not_called()

    def test_process_thinker_rejects_empty_or_oversized_prompts(self):
        for prompt in ("", "  \n ", "x" * 5001, None, 123):
            with self.subTest(prompt=prompt):
                response = self.client.post(
                    "/api/thinkers/process",
                    json={
                        "biography": "https://en.wikipedia.org/wiki/Carl_von_Clausewitz",
                        "prompt": prompt,
                    },
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    "Ollama prompt must be non-empty",
                    response.json["error"],
                )
        self.download_wikipedia_text.assert_not_called()
        self.extract_military_contributions.assert_not_called()

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_processing_existing_wikipedia_article_replaces_its_record(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        dbpedia_url = "https://dbpedia.org/data/Carl_von_Clausewitz.json"
        download_wikipedia.return_value = (b"<html>Original page</html>", wikipedia_url)
        download_dbpedia.return_value = (
            dbpedia_person("Carl_von_Clausewitz", "Carl von Clausewitz"),
            dbpedia_url,
        )

        created = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )
        original_wiki_document_id = created.json["wiki_document_id"]

        download_wikipedia.return_value = (b"<html>Updated page</html>", wikipedia_url)
        download_dbpedia.return_value = (
            dbpedia_person(
                "Carl_von_Clausewitz",
                "Carl von Clausewitz",
                "1780-07-01",
                "1831-11-16",
                "1795",
                "1831",
            ),
            dbpedia_url,
        )
        updated = self.client.post(
            "/api/thinkers/process",
            json={"biography": f"{wikipedia_url}/"},
        )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["id"], created.json["id"])
        self.assertEqual(
            updated.json["name"],
            "Carl von Clausewitz (1780-1831) (active 1795-1831)",
        )
        self.assertEqual(len(self.client.get("/api/thinkers").json), 1)
        self.assertEqual(len(self.mongo_client.documents.records), 4)
        self.assertEqual(
            self.client.get(
                f"/api/documents/{updated.json['wiki_document_id']}"
            ).get_data(),
            b"<html>Updated page</html>",
        )
        self.assertEqual(
            self.client.get(
                f"/api/documents/{original_wiki_document_id}"
            ).status_code,
            404,
        )

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_rejects_non_wikipedia_urls(
        self, download_wikipedia, download_dbpedia
    ):
        response = self.client.post(
            "/api/thinkers/process",
            json={"biography": "https://example.com/page"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("wikipedia.org", response.json["error"])
        download_wikipedia.assert_not_called()
        download_dbpedia.assert_not_called()
        self.assertEqual(self.client.get("/api/thinkers").json, [])

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_dbpedia_download_failure_does_not_create_partial_thinker(
        self, download_wikipedia, download_dbpedia
    ):
        download_wikipedia.return_value = (
            b"<html>Wikipedia article</html>",
            "https://en.wikipedia.org/wiki/Carl_von_Clausewitz",
        )
        download_dbpedia.side_effect = URLError("DBpedia is unavailable")

        response = self.client.post(
            "/api/thinkers/process",
            json={
                "biography": "https://en.wikipedia.org/wiki/Carl_von_Clausewitz",
            },
        )

        self.assertEqual(response.status_code, 502)
        self.assertIn("DBpedia", response.json["error"])
        self.assertEqual(self.client.get("/api/thinkers").json, [])
        self.assertEqual(self.mongo_client.documents.records, {})

    @patch("bruno.app._download_wikipedia_text", side_effect=URLError("Wikipedia API unavailable"))
    @patch("bruno.app._download_wikipedia_page")
    def test_wikipedia_plaintext_failure_does_not_create_partial_thinker(
        self, download_wikipedia, download_text
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        download_wikipedia.return_value = (b"<html>Clausewitz</html>", wikipedia_url)

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 502)
        self.assertIn("Could not retrieve Wikipedia plain text", response.json["error"])
        self.assertEqual(self.mongo_client.documents.records, {})
        self.assertEqual(self.client.get("/api/thinkers").json, [])
        download_text.assert_called_once_with(wikipedia_url)

    @patch("bruno.app._extract_military_contributions")
    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_language_model_failure_does_not_create_partial_thinker(
        self, download_wikipedia, download_dbpedia, extract_contributions
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        download_wikipedia.return_value = (b"<html>Clausewitz</html>", wikipedia_url)
        download_dbpedia.return_value = (
            dbpedia_person("Carl_von_Clausewitz", "Carl von Clausewitz"),
            "https://dbpedia.org/data/Carl_von_Clausewitz.json",
        )
        extract_contributions.side_effect = URLError("Ollama is unavailable")

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("language model is unavailable", response.json["error"])
        self.assertEqual(self.mongo_client.documents.records, {})
        self.assertEqual(self.client.get("/api/thinkers").json, [])

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_mongodb_failure_cleans_partial_documents_and_metadata(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"
        download_wikipedia.return_value = (b"<html>Clausewitz</html>", wikipedia_url)
        download_dbpedia.return_value = (
            dbpedia_person("Carl_von_Clausewitz", "Carl von Clausewitz"),
            "https://dbpedia.org/data/Carl_von_Clausewitz.json",
        )
        self.mongo_client.documents.fail_on_source = "military-strategy"

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("MongoDB", response.json["error"])
        self.assertEqual(self.mongo_client.documents.records, {})
        self.assertEqual(self.client.get("/api/thinkers").json, [])

    def test_missing_mongodb_documents_return_json_404(self):
        response = self.client.get(f"/api/documents/{ObjectId()}")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json, {"error": "Document not found."})

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_existing_thinker_updates_and_replaces_saved_documents(
        self, download_wikipedia, download_dbpedia
    ):
        created = self.client.post(
            "/api/thinkers",
            json={"name": "Old name", "biography": "https://en.wikipedia.org/wiki/Old"},
        )
        item_id = created.json["id"]
        updated_url = "https://en.wikipedia.org/wiki/New_Thinker"
        download_wikipedia.return_value = (b"<html>Updated article</html>", updated_url)
        dbpedia_url = "https://dbpedia.org/data/New_Thinker.json"
        dbpedia_content = dbpedia_person(
            "New_Thinker",
            "New Thinker",
            "1800-01-02",
            "1870-03-04",
            "1820",
            "1860",
        )
        download_dbpedia.return_value = (dbpedia_content, dbpedia_url)

        response = self.client.patch(
            f"/api/thinkers/{item_id}/process",
            json={"biography": updated_url},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["name"],
            "New Thinker (1800-1870) (active 1820-1860)",
        )
        self.assertEqual(response.json["biography"], updated_url)
        self.assertEqual(len(self.mongo_client.documents.records), 4)
        self.assertEqual(
            self.client.get(
                f"/api/documents/{response.json['wiki_document_id']}"
            ).get_data(),
            b"<html>Updated article</html>",
        )
        self.assertEqual(
            self.client.get(
                f"/api/documents/{response.json['dbpedia_document_id']}"
            ).get_data(),
            dbpedia_content,
        )

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_uses_dbpedia_name_without_active_dates(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Marie_Curie"
        download_wikipedia.return_value = (b"<html>Marie Curie</html>", wikipedia_url)
        dbpedia_url = "https://dbpedia.org/data/Marie_Curie.json"
        download_dbpedia.return_value = (
            dbpedia_person(
                "Marie_Curie",
                "Marie Curie",
                "1867-11-07",
                "1934-07-04",
                service_start=None,
                service_end=None,
            ),
            dbpedia_url,
        )

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json["name"],
            "Marie Curie (1867-1934)",
        )

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_extracts_year_from_month_prefixed_dbpedia_date(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Example_Thinker"
        download_wikipedia.return_value = (
            b"<html>Example Thinker</html>",
            wikipedia_url,
        )
        dbpedia_url = "https://dbpedia.org/data/Example_Thinker.json"
        download_dbpedia.return_value = (
            dbpedia_person(
                "Example_Thinker",
                "Example Thinker",
                "May 1633",
                "June 1701",
                service_start=None,
                service_end=None,
            ),
            dbpedia_url,
        )

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["name"], "Example Thinker (1633-1701)")
        dbpedia_view = self.client.get(
            f"/documents/{response.json['dbpedia_document_id']}"
        )
        self.assertIn(b"<dd>1633</dd>", dbpedia_view.data)
        self.assertNotIn(b"May 1633", dbpedia_view.data)

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_displays_bc_birth_and_death_years(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Socrates"
        download_wikipedia.return_value = (
            b"<html>Socrates (470 BC - 399 BC)</html>",
            wikipedia_url,
        )
        download_dbpedia.return_value = (
            dbpedia_person(
                "Socrates",
                "Socrates",
                "470",
                "399",
                service_start=None,
                service_end=None,
            ),
            "https://dbpedia.org/data/Socrates.json",
        )

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["name"], "Socrates (470-399 BC)")

    def test_dbpedia_year_is_bc_only_when_wikipedia_confirms_the_era(self):
        self.assertEqual(
            _dbpedia_year("544", b"<p>544&nbsp;BC - 496 BC</p>"),
            "544 BC",
        )
        self.assertEqual(_dbpedia_year("May 1633"), "1633")
        self.assertEqual(_dbpedia_year("544"), "544")

    @patch("bruno.app._download_dbpedia_data")
    @patch("bruno.app._download_wikipedia_page")
    def test_process_thinker_rejects_dbpedia_without_a_matching_name(
        self, download_wikipedia, download_dbpedia
    ):
        wikipedia_url = "https://en.wikipedia.org/wiki/Unknown_Person"
        download_wikipedia.return_value = (b"<html>Unknown</html>", wikipedia_url)
        download_dbpedia.return_value = (
            b'{"http://dbpedia.org/resource/Unknown_Person": {}}',
            "https://dbpedia.org/data/Unknown_Person.json",
        )

        response = self.client.post(
            "/api/thinkers/process", json={"biography": wikipedia_url}
        )

        self.assertEqual(response.status_code, 502)
        self.assertIn("name", response.json["error"])
        self.assertEqual(self.client.get("/api/thinkers").json, [])
        self.assertEqual(self.mongo_client.documents.records, {})

    def test_existing_thinker_database_is_migrated_for_mongo_documents(self):
        database_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            with connection:
                connection.execute(
                    "CREATE TABLE thinkers "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, biography TEXT)"
                )
                connection.execute(
                    "INSERT INTO thinkers (name, biography) VALUES (?, ?)",
                    ("Carl van Clausewitz", "https://en.wikipedia.org/wiki/Carl_von_Clausewitz"),
                )
        finally:
            connection.close()

        migrated_app = create_app(database_path, mongo_client=self.mongo_client)
        migrated_client = migrated_app.test_client()

        self.assertIsNone(
            migrated_client.get("/api/thinkers/1").json["wiki_document_id"]
        )
        self.assertIsNone(
            migrated_client.get("/api/thinkers/1").json["wiki_text_document_id"]
        )


if __name__ == "__main__":
    unittest.main()
