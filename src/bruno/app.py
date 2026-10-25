"""Web application and JSON API for topics and thinkers."""

from __future__ import annotations

import html
import json
import math
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from bson import ObjectId
from bson.errors import InvalidId
from flask import Flask, Response, jsonify, render_template, request
from markupsafe import Markup
from pymongo import MongoClient
from pymongo.errors import PyMongoError

EntitySpec = dict[str, Any]
MAX_DOWNLOAD_SIZE = 10 * 1024 * 1024
LLM_REQUEST_TIMEOUT = 900
DEFAULT_RESPONSE_WORD_COUNT = 500
MIN_RESPONSE_WORD_COUNT = 100
MAX_RESPONSE_WORD_COUNT = 1000
DEFAULT_OLLAMA_PROMPT = (
    "What are the thinker's contributions to military strategic thought?"
)
MAX_OLLAMA_PROMPT_LENGTH = 5000
REMOTE_REQUEST_TIMEOUT = 20
DBPEDIA_REQUEST_TIMEOUT = 120
WIKIPEDIA_HOSTS = ("wikipedia.org",)
DBPEDIA_HOSTS = ("dbpedia.org",)
_BERTOPIC_MODEL_LOCK = Lock()
_TOPIC_SEARCH_VECTORIZER_FILENAME = "topic_search_vectorizer.json"
_APPROXIMATE_DISTRIBUTIONS_FILENAME = "approximate_distributions.json"
_TOPIC_SOURCE_NAME_FILENAME = "topic_source.txt"
_TOPIC_SOURCE_DOCUMENTS_FILENAME = "topic_source_documents.json"
ENTITIES: dict[str, EntitySpec] = {
    "concepts": {
        "fields": {"name": ("TEXT", True), "description": ("TEXT", False)},
        "stored_fields": {"topic_assignments": "TEXT"},
        "label": "Topic",
    },
    "thinkers": {
        "fields": {"name": ("TEXT", True), "biography": ("TEXT", False)},
        "stored_fields": {
            "wiki_document_key": "TEXT",
            "wiki_document_id": "TEXT",
            "wiki_text_document_id": "TEXT",
            "dbpedia_document_id": "TEXT",
            "military_thinking_document_id": "TEXT",
        },
        "label": "Thinker",
    },
}


def create_app(
    database_path: str | Path | None = None,
    mongo_client: MongoClient | None = None,
) -> Flask:
    """Create the web application with SQLite metadata and MongoDB documents."""
    app = Flask(__name__)
    instance_path = Path(app.instance_path)
    instance_path.mkdir(parents=True, exist_ok=True)
    db_path = Path(
        database_path
        or os.environ.get("BRUNO_DATABASE", instance_path / "bruno.sqlite3")
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.config["DATABASE"] = str(db_path)
    app.config["BERTOPIC_MODEL_PATH"] = str(db_path.parent / "bertopic_model")
    client = mongo_client or MongoClient(
        os.environ.get("BRUNO_MONGODB_URI", "mongodb://localhost:27017"),
        serverSelectionTimeoutMS=3000,
    )
    app.extensions["bruno_mongo_client"] = client
    mongo_database = client[os.environ.get("BRUNO_MONGODB_DATABASE", "bruno")]
    app.config["MONGO_DOCUMENTS"] = mongo_database["documents"]
    _initialize_database(app.config["DATABASE"])

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api")
    def api_index():
        return jsonify(
            {
                "concepts": "/api/concepts",
                "generate_concepts": "/api/concepts/generate",
                "find_topics": "/api/concepts/find-topics",
                "thinkers": "/api/thinkers",
                "visualizations": "/visualizations/topics",
            }
        )

    @app.post("/api/concepts/find-topics")
    def find_topics():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        unknown_fields = set(payload) - {"concept"}
        if unknown_fields:
            return jsonify(
                {"error": f"Unknown field: {sorted(unknown_fields)[0]}."}
            ), 400
        concept = payload.get("concept")
        if (
            not isinstance(concept, str)
            or not concept.strip()
            or len(concept) > 500
        ):
            return jsonify(
                {
                    "error": (
                        "The search text must be a non-empty string of at most "
                        "500 characters."
                    )
                }
            ), 400
        with _database(app.config["DATABASE"]) as connection:
            concept_count = connection.execute(
                'SELECT COUNT(*) FROM "concepts"'
            ).fetchone()[0]
        if concept_count < 1:
            return jsonify(
                {"error": "Generate topics before searching related topics."}
            ), 409

        model_path = Path(app.config["BERTOPIC_MODEL_PATH"])
        if not model_path.is_dir():
            return jsonify(
                {"error": "Generate topics before searching BERTopic topics."}
            ), 409
        vectorizer_path = model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
        if not vectorizer_path.is_file():
            return jsonify(
                {
                    "error": (
                        "Regenerate topics to enable topic search for the "
                        "saved BERTopic model."
                    )
                }
            ), 409

        try:
            with _BERTOPIC_MODEL_LOCK:
                topic_model = _load_bertopic_model(model_path)
                _normalize_topic_embeddings(topic_model)
                if (
                    topic_model.topic_embeddings_ is None
                    or topic_model.topic_embeddings_.ndim != 2
                ):
                    raise ValueError(
                        "The saved BERTopic model does not contain searchable "
                        "topic embeddings."
                    )
                topic_model.embedding_model = _load_topic_search_embedder(
                    vectorizer_path
                )
                available_topics = len(topic_model.get_topics())
                if available_topics:
                    topic_ids, similarities = topic_model.find_topics(
                        search_term=concept.strip(),
                        top_n=min(concept_count, available_topics),
                    )
                else:
                    topic_ids, similarities = [], []
                if len(topic_ids) != len(similarities):
                    raise ValueError(
                        "BERTopic returned mismatched topic and similarity results."
                    )
                results = []
                for topic_id, similarity in zip(topic_ids, similarities):
                    topic_id = int(topic_id)
                    similarity = float(similarity)
                    if not math.isfinite(similarity):
                        raise ValueError(
                            "BERTopic returned an invalid topic similarity."
                        )
                    terms = topic_model.get_topic(topic_id) or []
                    name = " · ".join(term for term, _ in terms[:5] if term)
                    if not name:
                        name = (
                            "Unclassified topic"
                            if topic_id == -1
                            else f"Topic {topic_id}"
                        )
                    results.append(
                        {
                            "topic_id": topic_id,
                            "name": name,
                            "similarity": similarity,
                        }
                    )
        except ImportError as error:
            return jsonify(
                {"error": f"BERTopic is unavailable: {error}"}
            ), 503
        except ValueError as error:
            return jsonify(
                {"error": f"Could not find BERTopic topics: {error}"}
            ), 422
        except (OSError, RuntimeError) as error:
            return jsonify(
                {"error": f"Could not find BERTopic topics: {error}"}
            ), 500

        response = jsonify({"results": results})
        response.headers["Cache-Control"] = "no-store"
        return response

    def render_topic_visualization(visualization: str):
        model_path = Path(app.config["BERTOPIC_MODEL_PATH"])
        plot_explanation = None
        if not model_path.is_dir():
            return render_template(
                "visualization.html",
                plot_html=None,
                message="Generate topics before viewing visualizations.",
            )
        try:
            with _BERTOPIC_MODEL_LOCK:
                topic_model = _load_bertopic_model(model_path)
                if visualization in ("topics", "hierarchy", "heatmap"):
                    topic_ids = [
                        topic_id
                        for topic_id in topic_model.get_topics()
                        if topic_id != -1
                    ]
                    minimum_topic_count = 3 if visualization == "topics" else 2
                    if len(topic_ids) < minimum_topic_count:
                        if visualization == "topics":
                            message = (
                                "The intertopic distance map requires at least "
                                "three topics. Generate 3 or more topics to view it."
                            )
                        elif visualization == "hierarchy":
                            message = (
                                "Hierarchical clustering requires at least "
                                "two topics. Generate 2 or more topics to view it."
                            )
                        else:
                            message = (
                                "The topic similarity heatmap requires at least "
                                "two topics. Generate 2 or more topics to view it."
                            )
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=message,
                        )
                    if visualization == "topics":
                        figure = topic_model.visualize_topics(use_ctfidf=True)
                    elif visualization == "hierarchy":
                        documents_path = (
                            model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
                        )
                        if not documents_path.is_file():
                            return render_template(
                                "visualization.html",
                                plot_html=None,
                                message=(
                                    "Regenerate topics to save their source "
                                    "texts before viewing the hierarchy."
                                ),
                            )
                        documents = _load_topic_source_documents(documents_path)
                        try:
                            from nltk.tokenize import sent_tokenize

                            sentence_documents, _, _ = (
                                _split_documents_into_sentences(
                                    [document["text"] for document in documents],
                                    [document["name"] for document in documents],
                                    list(range(len(documents))),
                                    sent_tokenize,
                                )
                            )
                        except LookupError as error:
                            raise RuntimeError(
                                "NLTK sentence tokenizer data is missing. "
                                "Install it with "
                                "`python -m nltk.downloader punkt_tab`."
                            ) from error
                        hierarchical_topics = topic_model.hierarchical_topics(
                            sentence_documents
                        )
                        figure = topic_model.visualize_hierarchy(
                            hierarchical_topics=hierarchical_topics
                        )
                    else:
                        figure = topic_model.visualize_heatmap()
                elif visualization == "topics-over-time":
                    documents_path = (
                        model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
                    )
                    if not documents_path.is_file():
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Regenerate topics to save their source texts "
                                "before viewing topics over time."
                            ),
                        )
                    documents = _load_topic_source_documents(documents_path)
                    timestamps = [
                        _approximate_thinker_year(document["name"])
                        for document in documents
                    ]
                    if any(timestamp is None for timestamp in timestamps):
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Topics over time requires an approximate year "
                                "for every thinker. Add thinker dates and "
                                "regenerate topics."
                            ),
                        )
                    timestamps_by_thinker = {
                        index: timestamp
                        for index, timestamp in enumerate(timestamps)
                        if timestamp is not None
                    }
                    try:
                        from nltk.tokenize import sent_tokenize

                        sentence_documents, _, sentence_thinker_ids = (
                            _split_documents_into_sentences(
                                [document["text"] for document in documents],
                                [document["name"] for document in documents],
                                list(range(len(documents))),
                                sent_tokenize,
                            )
                        )
                    except LookupError as error:
                        raise RuntimeError(
                            "NLTK sentence tokenizer data is missing. Install "
                            "it with `python -m nltk.downloader punkt_tab`."
                        ) from error
                    sentence_timestamps = [
                        max(1800, timestamps_by_thinker[thinker_id])
                        for thinker_id in sentence_thinker_ids
                    ]
                    time_topics = topic_model.topics_over_time(
                        sentence_documents,
                        sentence_timestamps,
                    )
                    figure = topic_model.visualize_topics_over_time(time_topics)
                    plot_explanation = (
                        "For readability, all approximate years before 1800 "
                        "are displayed as 1800."
                    )
                elif visualization == "topics-per-class":
                    documents_path = (
                        model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
                    )
                    if not documents_path.is_file():
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Regenerate topics to save their source texts "
                                "before viewing topics per thinker."
                            ),
                        )
                    documents = _load_topic_source_documents(documents_path)
                    try:
                        from nltk.tokenize import sent_tokenize

                        sentence_documents, sentence_thinker_names, _ = (
                            _split_documents_into_sentences(
                                [document["text"] for document in documents],
                                [document["name"] for document in documents],
                                list(range(len(documents))),
                                sent_tokenize,
                            )
                        )
                    except LookupError as error:
                        raise RuntimeError(
                            "NLTK sentence tokenizer data is missing. Install "
                            "it with `python -m nltk.downloader punkt_tab`."
                        ) from error
                    class_names = [
                        re.sub(r"(?:\s+\([^)]*\))+$", "", name)
                        for name in sentence_thinker_names
                    ]
                    class_years = {
                        re.sub(r"(?:\s+\([^)]*\))+$", "", document["name"]): (
                            _approximate_thinker_year(document["name"])
                        )
                        for document in documents
                    }
                    class_order = sorted(
                        class_years,
                        key=lambda name: (
                            class_years[name] is None,
                            class_years[name] or 0,
                        ),
                    )
                    topics_per_class = topic_model.topics_per_class(
                        sentence_documents,
                        class_names,
                    )
                    figure = topic_model.visualize_topics_per_class(
                        topics_per_class
                    )
                    figure.update_yaxes(
                        categoryorder="array",
                        categoryarray=class_order,
                        autorange="reversed",
                    )
                elif visualization in (
                    "visualize_documents",
                    "hierarchical_documents",
                ):
                    documents_path = (
                        model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
                    )
                    if not documents_path.is_file():
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Regenerate topics to save their source texts "
                                "before visualizing documents."
                            ),
                        )
                    # Load saved documents; expect each item to contain at least 'text' and optionally 'topic'
                    documents = _load_topic_source_documents(documents_path)
                    docs_texts = [doc.get("text", "") for doc in documents]
                    # If no documents or texts, show message
                    if not any(docs_texts):
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "No saved document texts available to visualize."
                            ),
                        )
                    # Use BERTopic visualize_documents (returns a plotly figure)
                    # The model is trained on sentences, so visualize the same units.
                    try:
                        from nltk.tokenize import sent_tokenize

                        sentence_documents, _, _ = (
                            _split_documents_into_sentences(
                                docs_texts,
                                [doc["name"] for doc in documents],
                                list(range(len(documents))),
                                sent_tokenize,
                            )
                        )
                    except LookupError as error:
                        raise RuntimeError(
                            "NLTK sentence tokenizer data is missing. Install "
                            "it with `python -m nltk.downloader punkt_tab`."
                        ) from error
                    if visualization == "hierarchical_documents":
                        if len(topic_model.get_topics()) - (
                            1 if -1 in topic_model.get_topics() else 0
                        ) < 2:
                            return render_template(
                                "visualization.html",
                                plot_html=None,
                                message=(
                                    "Hierarchical thinkers and topics require "
                                    "at least two topics. Generate 2 or more "
                                    "topics to view it."
                                ),
                            )
                        hierarchical_topics = topic_model.hierarchical_topics(
                            sentence_documents
                        )
                        figure = topic_model.visualize_hierarchical_documents(
                            sentence_documents,
                            hierarchical_topics,
                            nr_levels=max(1, min(10, len(hierarchical_topics))),
                            hide_document_hover=True,
                        )
                    else:
                        figure = topic_model.visualize_documents(
                            sentence_documents,
                            hide_document_hover=True,
                        )
                    figure.update_layout(width=None, autosize=True)
                    plot_html = Markup(
                        figure.to_html(
                            full_html=False,
                            include_plotlyjs="cdn",
                            config={"responsive": True},
                        )
                    )
                else:
                    documents_path = (
                        model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
                    )
                    if not documents_path.is_file():
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Regenerate topics to save their source texts "
                                "before viewing approximate distribution."
                            ),
                        )
                    if not (
                        model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
                    ).is_file():
                        return render_template(
                            "visualization.html",
                            plot_html=None,
                            message=(
                                "Regenerate topics to save their text "
                                "embedding vectorizer before viewing "
                                "approximate distribution."
                            ),
                        )
                    documents = _load_topic_source_documents(documents_path)
                    distribution_html = []
                    distributions = _calculate_approximate_distributions(
                        topic_model,
                        documents,
                    )
                    for distribution in distributions:
                        document = distribution["document"]
                        table = topic_model.visualize_approximate_distribution(
                            document["text"],
                            distribution["token_distribution"],
                        )
                        distribution_html.append(
                            '<section class="approximate-distribution" '
                            f'data-search-text="{html.escape(document["name"].casefold(), quote=True)}">'
                            f'<h2>{html.escape(document["name"])}</h2>'
                            f'{table.format(escape="html").to_html()}'
                            "</section>"
                        )
                    plot_html = Markup("\n".join(distribution_html))
                if visualization in (
                    "topics",
                    "hierarchy",
                    "heatmap",
                    "topics-over-time",
                    "topics-per-class",
                ):
                    figure.update_layout(width=None, autosize=True)
                    plot_html = Markup(
                        figure.to_html(
                            full_html=False,
                            include_plotlyjs="cdn",
                            config={"responsive": True},
                        )
                    )
        except ImportError as error:
            return (
                render_template(
                    "visualization.html",
                    plot_html=None,
                    error=f"The visualization engine is unavailable: {error}",
                ),
                503,
            )
        except (OSError, ValueError, RuntimeError) as error:
            return (
                render_template(
                    "visualization.html",
                    plot_html=None,
                    error=f"Could not load visualization: {error}",
                ),
                500,
            )
        response = app.make_response(
            render_template(
                "visualization.html",
                plot_html=plot_html,
                plot_explanation=plot_explanation,
            )
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/visualizations/topics")
    def visualize_topics():
        return render_topic_visualization("topics")

    @app.get("/visualizations/heatmap")
    def visualize_heatmap():
        return render_topic_visualization("heatmap")

    @app.get("/visualizations/topics-over-time")
    def visualize_topics_over_time():
        return render_topic_visualization("topics-over-time")

    @app.get("/visualizations/topics-per-class")
    def visualize_topics_per_class():
        return render_topic_visualization("topics-per-class")

    @app.get("/visualizations/approximate-distribution")
    def visualize_approximate_distribution():
        return render_topic_visualization("approximate_distribution")

    @app.get("/api/visualizations/approximate-distributions")
    def approximate_distributions():
        model_path = Path(app.config["BERTOPIC_MODEL_PATH"])
        documents_path = model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
        vectorizer_path = model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
        if not model_path.is_dir() or not documents_path.is_file():
            return jsonify(
                {"error": "Generate topics before viewing approximate distributions."}
            ), 409
        if not vectorizer_path.is_file():
            return jsonify(
                {
                    "error": (
                        "Regenerate topics to save their text embedding "
                        "vectorizer before viewing approximate distributions."
                    )
                }
            ), 409
        try:
            documents = _load_topic_source_documents(documents_path)
            with _BERTOPIC_MODEL_LOCK:
                topic_model = _load_bertopic_model(model_path)
                distributions = _approximate_distributions(
                    topic_model,
                    documents,
                    model_path,
                )
            source_path = model_path / _TOPIC_SOURCE_NAME_FILENAME
            saved_source = (
                source_path.read_text(encoding="utf-8").strip()
                if source_path.is_file()
                else ""
            )
            id_field = (
                "wiki_text_document_id"
                if saved_source == "wikipedia"
                else "military_thinking_document_id"
            )
            with _database(app.config["DATABASE"]) as connection:
                document_ids = {
                    row["name"]: row[id_field]
                    for row in connection.execute(
                        'SELECT name, wiki_text_document_id, '
                        'military_thinking_document_id FROM "thinkers"'
                    ).fetchall()
                }
            topics = _format_approximate_topic_distributions(
                topic_model,
                distributions,
                document_ids,
            )
        except ImportError as error:
            return jsonify(
                {"error": f"The visualization engine is unavailable: {error}"}
            ), 503
        except (OSError, ValueError, RuntimeError) as error:
            return jsonify(
                {"error": f"Could not load approximate distributions: {error}"}
            ), 500
        response = jsonify({"topics": topics})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/visualizations/hierarchy")
    def visualize_hierarchy():
        return render_topic_visualization("hierarchy")

    @app.get("/visualizations/visualize-documents")
    def visualize_documents():
        return render_topic_visualization("visualize_documents")

    @app.get("/visualizations/hierarchical-documents")
    def visualize_hierarchical_documents():
        return render_topic_visualization("hierarchical_documents")

    @app.get("/api/status/language-model")
    def language_model_status():
        return jsonify({"connected": _language_model_connected()})

    @app.post("/api/concepts/generate")
    def generate_concepts():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        unknown_fields = set(payload) - {
            "topic_count",
            "topic_source",
            "stop_words",
            "min_df",
            "max_words_in_topic",
            "top_n_words",
            "diversity",
        }
        if unknown_fields:
            return jsonify(
                {"error": f"Unknown field: {sorted(unknown_fields)[0]}."}
            ), 400
        topic_count = payload.get("topic_count")
        if topic_count is not None and (
            type(topic_count) is not int or topic_count < 1
        ):
            return jsonify(
                {"error": "topic_count must be an integer of at least 1."}
            ), 400
        topic_source = payload.get("topic_source", "wikipedia")
        if not isinstance(topic_source, str) or topic_source not in {
            "ollama",
            "wikipedia",
        }:
            return jsonify(
                {"error": "topic_source must be 'ollama' or 'wikipedia'."}
            ), 400
        stop_words = payload.get("stop_words", "english")
        if stop_words is not None and stop_words != "english":
            return jsonify(
                {"error": "stop_words must be 'english' or null."}
            ), 400
        min_df = payload.get("min_df", 3)
        if type(min_df) is not int or min_df < 1:
            return jsonify(
                {"error": "min_df must be an integer of at least 1."}
            ), 400
        max_words_in_topic = payload.get("max_words_in_topic", 3)
        if type(max_words_in_topic) is not int or max_words_in_topic < 1:
            return jsonify(
                {
                    "error": (
                        "max_words_in_topic must be an integer of at least 1."
                    )
                }
            ), 400
        top_n_words = payload.get("top_n_words", 15)
        if type(top_n_words) is not int or top_n_words < 1:
            return jsonify(
                {"error": "top_n_words must be an integer of at least 1."}
            ), 400
        diversity = payload.get("diversity", 1.0)
        if (
            type(diversity) not in (int, float)
            or not 0 <= diversity <= 1
        ):
            return jsonify(
                {"error": "diversity must be a number between 0 and 1."}
            ), 400
        vectorizer_ngram_range = (1, max_words_in_topic)
        source_config = {
            "ollama": {
                "document_id_field": "military_thinking_document_id",
                "document_source": "military-strategy",
                "label": "Ollama analysis",
                "plural_label": "Ollama analyses",
            },
            "wikipedia": {
                "document_id_field": "wiki_text_document_id",
                "document_source": "wikipedia-text",
                "label": "Wikipedia text",
                "plural_label": "Wikipedia texts",
            },
        }[topic_source]
        source_label = source_config["label"]
        source_plural_label = source_config["plural_label"]
        document_source = source_config["document_source"]

        with _database(app.config["DATABASE"]) as connection:
            thinkers = connection.execute(
                'SELECT id, name, biography, wiki_text_document_id, '
                'military_thinking_document_id, wiki_document_key '
                'FROM "thinkers" ORDER BY id'
            ).fetchall()
        if not thinkers:
            return jsonify(
                {"error": "Process at least one thinker before generating topics."}
            ), 409

        documents = app.config["MONGO_DOCUMENTS"]
        source_texts: list[str] = []
        thinker_names: list[str] = []
        thinker_ids: list[int] = []
        missing_texts: list[str] = []
        fetched_sources: list[dict[str, Any]] = []
        try:
            for thinker in thinkers:
                document = None
                document_id = thinker[source_config["document_id_field"]]
                if document_id:
                    try:
                        document = documents.find_one(
                            {
                                "_id": ObjectId(document_id),
                                "source": document_source,
                            }
                        )
                    except InvalidId:
                        document = None
                    if (
                        document is not None
                        and document.get("source") != document_source
                    ):
                        document = None
                if document is None and thinker["wiki_document_key"]:
                    document = documents.find_one(
                        {
                            "document_key": thinker["wiki_document_key"],
                            "source": document_source,
                        }
                    )
                if (
                    document is None
                    and topic_source == "wikipedia"
                    and thinker["biography"]
                ):
                    try:
                        source_text = _download_wikipedia_text(thinker["biography"])
                    except (HTTPError, URLError, TimeoutError, OSError) as error:
                        return jsonify(
                            {
                                "error": (
                                    f"Could not retrieve Wikipedia text for "
                                    f"{thinker['name']}: {error}"
                                )
                            }
                        ), 502
                    except ValueError as error:
                        return jsonify(
                            {
                                "error": (
                                    f"Could not retrieve Wikipedia text for "
                                    f"{thinker['name']}: {error}"
                                )
                            }
                        ), 422
                    if source_text.strip():
                        fetched_sources.append(
                            {
                                "id": thinker["id"],
                                "name": thinker["name"],
                                "url": thinker["biography"],
                                "document_key": thinker["wiki_document_key"]
                                or uuid.uuid4().hex,
                                "content": source_text.strip().encode("utf-8"),
                            }
                        )
                        thinker_ids.append(thinker["id"])
                        thinker_names.append(thinker["name"])
                        source_texts.append(source_text.strip())
                        continue
                if (
                    document is None
                    or document.get("source") != document_source
                    or not isinstance(document.get("content"), bytes)
                ):
                    missing_texts.append(thinker["name"])
                    continue
                source_text = document["content"].decode("utf-8").strip()
                if not source_text:
                    missing_texts.append(thinker["name"])
                    continue
                thinker_ids.append(thinker["id"])
                thinker_names.append(thinker["name"])
                source_texts.append(source_text)
        except PyMongoError as error:
            return jsonify(
                {"error": f"MongoDB is unavailable: {error}"}
            ), 503
        except UnicodeDecodeError as error:
            return jsonify(
                {"error": f"A saved {source_label} is not valid UTF-8: {error}"}
            ), 422
        if missing_texts:
            return jsonify(
                {
                    "error": (
                        f"Reprocess these thinkers to retrieve their {source_label} "
                        f"before generating topics: {', '.join(missing_texts)}."
                    )
                }
            ), 409
        if topic_count is not None and topic_count > len(source_texts):
            return jsonify(
                {
                    "error": (
                        "The requested number of topics cannot exceed the "
                        f"number of thinker texts ({len(source_texts)})."
                    )
                }
            ), 422

        with _database(app.config["DATABASE"]) as connection:
            connection.execute('DELETE FROM "concept_thinkers"')
            connection.execute('DELETE FROM "concepts"')

        try:
            concepts = _generate_bertopic_concepts(
                source_texts,
                thinker_names,
                topic_count,
                thinker_ids,
                model_path=Path(app.config["BERTOPIC_MODEL_PATH"]),
                stop_words=stop_words,
                min_df=min_df,
                ngram_range=vectorizer_ngram_range,
                top_n_words=top_n_words,
                diversity=diversity,
            )
        except ImportError as error:
            return jsonify(
                {"error": f"BERTopic is unavailable: {error}"}
            ), 503
        except (ValueError, RuntimeError) as error:
            return jsonify(
                {"error": f"BERTopic could not generate topics: {error}"}
            ), 422
        except OSError as error:
            return jsonify(
                {"error": f"Could not save the BERTopic model: {error}"}
            ), 500
        if not concepts:
            return jsonify(
                {
                    "error": (
                        f"BERTopic did not find any topics in the selected "
                        f"{source_plural_label.lower()}."
                    )
                }
            ), 422

        inserted_source_ids = []
        created_concepts = []
        try:
            (
                Path(app.config["BERTOPIC_MODEL_PATH"])
                / _TOPIC_SOURCE_NAME_FILENAME
            ).write_text(topic_source, encoding="utf-8")
        except OSError:
            pass
        try:
            for source in fetched_sources:
                source_id = _store_document(
                    documents,
                    source["document_key"],
                    "wikipedia-text",
                    source["url"],
                    source["content"],
                    "text/plain; charset=utf-8",
                )
                inserted_source_ids.append(source_id)
                source["document_id"] = source_id
            with _database(app.config["DATABASE"]) as connection:
                for source in fetched_sources:
                    connection.execute(
                        'UPDATE "thinkers" SET wiki_document_key = ?, '
                        "wiki_text_document_id = ? WHERE id = ?",
                        (
                            source["document_key"],
                            source["document_id"],
                            source["id"],
                        ),
                    )
                for concept in concepts:
                    cursor = connection.execute(
                        'INSERT INTO "concepts" '
                        "(name, description, topic_assignments) VALUES (?, ?, ?)",
                        (
                            concept["name"],
                            "",
                            json.dumps(
                                [
                                    {**assignment, "topic_source": topic_source}
                                    for assignment in concept.get(
                                        "topic_assignments", []
                                    )
                                ]
                            ),
                        ),
                    )
                    concept_id = cursor.lastrowid
                    for thinker_id in concept["thinker_ids"]:
                        connection.execute(
                            'INSERT INTO "concept_thinkers" '
                            "(concept_id, thinker_id) VALUES (?, ?)",
                            (concept_id, thinker_id),
                        )
                    row = connection.execute(
                        'SELECT * FROM "concepts" WHERE id = ?', (concept_id,)
                    ).fetchone()
                    created_concepts.append(
                        _serialize_concept(row, connection)
                    )
        except (PyMongoError, sqlite3.Error) as error:
            if inserted_source_ids:
                try:
                    documents.delete_many(
                        {
                            "_id": {
                                "$in": [
                                    ObjectId(source_id)
                                    for source_id in inserted_source_ids
                                ]
                            }
                        }
                    )
                except PyMongoError as cleanup_error:
                    return jsonify(
                        {
                            "error": (
                                f"Could not save generated topics: {error}; "
                                "cleanup of fetched Wikipedia texts also failed: "
                                f"{cleanup_error}"
                            )
                        }
                    ), 503
            return jsonify(
                {"error": f"Could not save generated topics: {error}"}
            ), 503
        return jsonify(
            {
                "concepts": created_concepts,
                "algorithm": "BERTopic",
                "source_thinkers": len(thinker_names),
                "topic_source": topic_source,
            }
        ), 201

    @app.get("/api/documents/<document_id>")
    def saved_document(document_id: str):
        try:
            mongo_id = ObjectId(document_id)
        except InvalidId:
            return jsonify({"error": "Document not found."}), 404
        try:
            document = app.config["MONGO_DOCUMENTS"].find_one({"_id": mongo_id})
        except PyMongoError as error:
            return jsonify({"error": f"MongoDB is unavailable: {error}"}), 503
        if document is None:
            return jsonify({"error": "Document not found."}), 404
        return Response(
            document["content"],
            mimetype=document["content_type"],
            headers={"Content-Disposition": "inline"},
        )

    @app.put("/api/documents/<document_id>")
    def update_saved_document(document_id: str):
        try:
            mongo_id = ObjectId(document_id)
        except InvalidId:
            return jsonify({"error": "Document not found."}), 404
        payload = request.get_json(silent=True)
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "The text must be a non-empty string."}), 400
        try:
            result = app.config["MONGO_DOCUMENTS"].update_one(
                {"_id": mongo_id, "source": "wikipedia-text"},
                {"$set": {"content": text.strip().encode("utf-8")}},
            )
        except PyMongoError as error:
            return jsonify({"error": f"MongoDB is unavailable: {error}"}), 503
        if result.matched_count == 0:
            return jsonify({"error": "Document not found."}), 404
        return jsonify({"saved": True})

    @app.get("/documents/<document_id>")
    def document_view(document_id: str):
        try:
            mongo_id = ObjectId(document_id)
        except InvalidId:
            return jsonify({"error": "Document not found."}), 404
        try:
            document = app.config["MONGO_DOCUMENTS"].find_one({"_id": mongo_id})
        except PyMongoError as error:
            return jsonify({"error": f"MongoDB is unavailable: {error}"}), 503
        if document is None:
            return jsonify({"error": "Document not found."}), 404
        try:
            view_data = _document_view_data(
                document, app.config["MONGO_DOCUMENTS"]
            )
        except PyMongoError as error:
            return jsonify({"error": f"MongoDB is unavailable: {error}"}), 503
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            return jsonify({"error": f"Could not display saved document: {error}"}), 422
        view_data["document_id"] = document_id
        return render_template("document.html", **view_data)

    for entity, spec in ENTITIES.items():
        _register_entity_routes(app, entity, spec)

    return app


def _generate_bertopic_concepts(
    documents: list[str],
    thinker_names: list[str],
    requested_topic_count: int | None = None,
    thinker_ids: list[int] | None = None,
    model_path: Path | None = None,
    stop_words: str | None = "english",
    min_df: int = 3,
    ngram_range: tuple[int, int] = (1, 5),
    top_n_words: int = 15,
    diversity: float = 1.0,
) -> list[dict[str, Any]]:
    import numpy as np
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from nltk.tokenize import sent_tokenize
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import FunctionTransformer

    if not documents:
        raise ValueError("At least one source text is required.")
    if thinker_ids is None:
        thinker_ids = list(range(len(thinker_names)))

    try:
        sentence_documents, _, _ = (
            _split_documents_into_sentences(
                documents,
                thinker_names,
                thinker_ids,
                sent_tokenize,
            )
        )
    except LookupError as error:
        raise RuntimeError(
            "NLTK sentence tokenizer data is missing. Install it with "
            "`python -m nltk.downloader punkt_tab`."
        ) from error

    embedding_model_name = _embedding_model_name()
    embeddings = _get_sentence_embedder(embedding_model_name).encode(
        sentence_documents,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    document_count, feature_count = embeddings.shape

    if requested_topic_count is None:
        topic_count = max(
            1, min(document_count, round(math.sqrt(document_count / 2)))
        )
    elif requested_topic_count > document_count:
        raise ValueError(
            "The requested number of topics cannot exceed the number of "
            f"sentences ({document_count})."
        )
    else:
        topic_count = requested_topic_count
    dimensionality = (
        FunctionTransformer()
        if document_count == 1
        else TruncatedSVD(
            n_components=min(10, document_count, feature_count),
            random_state=42,
        )
    )
    class GaussianMixtureClusterer:
        def __init__(self, components: int):
            self.model = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=42,
                n_init=5,
            )

        def fit(self, values: Any, y: Any = None) -> GaussianMixtureClusterer:
            if hasattr(values, "toarray"):
                values = values.toarray()
            if values.shape[0] == 1:
                self.labels_ = np.zeros(1, dtype=int)
                self.probabilities_ = np.ones((1, 1), dtype=float)
                return self
            self.model.fit(values)
            self.labels_ = self.model.predict(values)
            self.probabilities_ = self.model.predict_proba(values)
            return self

    topic_model = BERTopic(
        embedding_model=_sentence_transformer_backend(embedding_model_name),
        umap_model=dimensionality,
        hdbscan_model=GaussianMixtureClusterer(topic_count),
        vectorizer_model=CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df,
        ),
        top_n_words=top_n_words,
        representation_model=MaximalMarginalRelevance(diversity=diversity),
        calculate_probabilities=True,
        verbose=False,
    )
    topic_model.fit_transform(sentence_documents, embeddings=embeddings)
    _normalize_topic_embeddings(topic_model)
    source_documents = [
        {"name": name, "text": text}
        for name, text in zip(thinker_names, documents)
    ]
    if model_path is not None:
        with _BERTOPIC_MODEL_LOCK:
            (model_path / _APPROXIMATE_DISTRIBUTIONS_FILENAME).unlink(
                missing_ok=True
            )
            topic_model.save(
                str(model_path),
                serialization="safetensors",
                save_embedding_model=False,
                save_ctfidf=True,
            )
            _save_topic_embedding_config(model_path, embedding_model_name)
            _save_topic_source_documents(model_path, documents, thinker_names)
            distributions = _approximate_distributions(
                topic_model, source_documents, model_path
            )
    else:
        distributions = _calculate_approximate_distributions(
            topic_model, source_documents
        )

    return _concepts_from_distributions(
        distributions,
        thinker_ids,
        topic_model.get_topic,
    )


def _split_documents_into_sentences(
    documents: list[str],
    thinker_names: list[str],
    thinker_ids: list[int],
    sentence_tokenizer: Callable[[str], list[str]],
) -> tuple[list[str], list[str], list[int]]:
    if len(documents) != len(thinker_names) or len(documents) != len(thinker_ids):
        raise ValueError(
            "Source texts, thinker names, and thinker IDs must have matching lengths."
        )

    sentence_documents: list[str] = []
    sentence_names: list[str] = []
    sentence_thinker_ids: list[int] = []
    for document, thinker_name, thinker_id in zip(
        documents,
        thinker_names,
        thinker_ids,
    ):
        sentences = [
            sentence.strip()
            for sentence in sentence_tokenizer(document)
            if sentence.strip()
        ]
        if not sentences:
            raise ValueError(
                f"Could not split the source text for {thinker_name} into sentences."
            )
        sentence_documents.extend(sentences)
        sentence_names.extend([thinker_name] * len(sentences))
        sentence_thinker_ids.extend([thinker_id] * len(sentences))
    return sentence_documents, sentence_names, sentence_thinker_ids


def _concepts_from_distributions(
    distributions: list[dict[str, Any]],
    thinker_ids: list[int],
    get_topic: Callable[[int], Any],
) -> list[dict[str, Any]]:
    if not distributions or len(distributions) != len(thinker_ids):
        raise ValueError(
            "BERTopic returned approximate distributions for a different "
            "number of source documents."
        )
    topic_ids = distributions[0]["topic_ids"]

    # A thinker belongs to the topic with the highest approximate share.
    members: dict[int, list[int]] = {topic_id: [] for topic_id in topic_ids}
    for index, distribution in enumerate(distributions):
        probabilities = distribution["probabilities"]
        strongest = max(range(len(topic_ids)), key=probabilities.__getitem__)
        members[topic_ids[strongest]].append(index)
    # Topics that are nobody's strongest still list their best-matching thinker.
    for position, topic_id in enumerate(topic_ids):
        if not members[topic_id]:
            members[topic_id].append(
                max(
                    range(len(distributions)),
                    key=lambda index: distributions[index]["probabilities"][
                        position
                    ],
                )
            )

    concepts: list[dict[str, Any]] = []
    for position, topic_id in enumerate(topic_ids):
        terms = get_topic(topic_id) or []
        keywords = [term for term, _ in terms[:5] if term]
        concepts.append(
            {
                "name": " · ".join(keywords) if keywords else f"Topic {topic_id}",
                "thinker_ids": [thinker_ids[index] for index in members[topic_id]],
                "topic_assignments": [
                    {
                        "thinker_id": thinker_ids[index],
                        "topic_id": topic_id,
                        "probability": distributions[index]["probabilities"][
                            position
                        ],
                    }
                    for index in members[topic_id]
                ],
            }
        )
    return concepts


def _approximate_distributions(
    topic_model: Any,
    documents: list[dict[str, str]],
    model_path: Path,
) -> list[dict[str, Any]]:
    """Return the approximate distributions, calculating them only once.

    The Topics tab and the Visualizations legend share this cached result.
    """
    cache_path = model_path / _APPROXIMATE_DISTRIBUTIONS_FILENAME
    names = [document["name"] for document in documents]
    topic_ids = sorted(
        topic_id for topic_id in topic_model.get_topics() if topic_id >= 0
    )
    cached = _read_approximate_distributions(cache_path, names, topic_ids)
    if cached is not None:
        return cached
    distributions = _calculate_approximate_distributions(topic_model, documents)
    if distributions:
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "topic_ids": topic_ids,
                    "distributions": [
                        {
                            "name": distribution["document"]["name"],
                            "probabilities": distribution["probabilities"],
                        }
                        for distribution in distributions
                    ],
                }
            ),
            encoding="utf-8",
        )
        temporary_path.replace(cache_path)
    return distributions


def _read_approximate_distributions(
    cache_path: Path,
    names: list[str],
    topic_ids: list[int],
) -> list[dict[str, Any]] | None:
    try:
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        entries = saved["distributions"]
        if (
            saved["topic_ids"] != topic_ids
            or [entry["name"] for entry in entries] != names
            or any(
                len(entry["probabilities"]) != len(topic_ids)
                for entry in entries
            )
        ):
            return None
        return [
            {
                "document": {"name": entry["name"]},
                "topic_ids": topic_ids,
                "probabilities": [float(v) for v in entry["probabilities"]],
            }
            for entry in entries
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None

def _load_bertopic_model(model_path: Path) -> Any:
    from bertopic import BERTopic

    vectorizer_path = model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
    if vectorizer_path.is_file():
        return BERTopic.load(
            str(model_path),
            embedding_model=_load_topic_embedding_model(vectorizer_path),
        )
    return BERTopic.load(str(model_path))


def _normalize_topic_embeddings(topic_model: Any) -> None:
    embeddings = topic_model.topic_embeddings_
    if embeddings is not None and embeddings.ndim == 3 and embeddings.shape[1] == 1:
        topic_model.topic_embeddings_ = embeddings[:, 0, :]


_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def _embedding_model_name() -> str:
    return os.environ.get("BRUNO_EMBEDDING_MODEL") or _DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=2)
def _get_sentence_embedder(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _save_topic_embedding_config(model_path: Path, model_name: str) -> None:
    config_path = model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
    temporary_path = config_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"embedding_model": model_name}), encoding="utf-8"
    )
    temporary_path.replace(config_path)


def _sentence_transformer_backend(model_name: Any) -> Any:
    from bertopic.backend._utils import select_backend

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("The saved topic embedding model is invalid.")
    return select_backend(_get_sentence_embedder(model_name))


def _save_topic_search_vectorizer(
    model_path: Path,
    vectorizer: Any,
) -> None:
    vectorizer_path = model_path / _TOPIC_SEARCH_VECTORIZER_FILENAME
    temporary_path = vectorizer_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "vocabulary": {
                    term: int(index)
                    for term, index in vectorizer.vocabulary_.items()
                },
                "idf": vectorizer.idf_.tolist(),
                "ngram_range": list(vectorizer.get_params()["ngram_range"]),
                "stop_words": vectorizer.get_params()["stop_words"],
            }
        ),
        encoding="utf-8",
    )
    temporary_path.replace(vectorizer_path)


def _load_topic_embedding_model(vectorizer_path: Path) -> Any:
    import numpy as np
    from bertopic.backend import BaseEmbedder
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

    try:
        saved_vectorizer = json.loads(vectorizer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("The saved topic-search vectorizer is invalid.") from error

    if not isinstance(saved_vectorizer, dict):
        raise ValueError("The saved topic-search vectorizer is invalid.")
    if "embedding_model" in saved_vectorizer:
        return _sentence_transformer_backend(saved_vectorizer["embedding_model"])
    vocabulary = saved_vectorizer.get("vocabulary")
    idf = saved_vectorizer.get("idf")
    ngram_range = saved_vectorizer.get("ngram_range", [1, 5])
    stop_words = saved_vectorizer.get("stop_words", "english")
    if (
        not isinstance(vocabulary, dict)
        or not isinstance(idf, list)
        or not isinstance(ngram_range, list)
        or len(ngram_range) != 2
        or any(type(value) is not int or value < 1 for value in ngram_range)
        or ngram_range[0] > ngram_range[1]
        or stop_words not in ("english", None)
        or not vocabulary
        or len(vocabulary) != len(idf)
        or any(
            not isinstance(term, str)
            or type(index) is not int
            or index < 0
            or index >= len(idf)
            for term, index in vocabulary.items()
        )
        or set(vocabulary.values()) != set(range(len(idf)))
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            for value in idf
        )
    ):
        raise ValueError("The saved topic-search vectorizer is invalid.")

    vectorizer = TfidfVectorizer(
        vocabulary=vocabulary,
        ngram_range=tuple(ngram_range),
        stop_words=stop_words,
        max_features=20000,
    )
    vectorizer.vocabulary_ = vocabulary
    vectorizer.fixed_vocabulary_ = True
    vectorizer._tfidf = TfidfTransformer()
    vectorizer._tfidf.idf_ = np.asarray(idf)

    class TfidfTopicEmbedder(BaseEmbedder):
        def __init__(self) -> None:
            super().__init__()

        def embed(self, documents: list[str], verbose: bool = False) -> Any:
            return vectorizer.transform(documents)

    return TfidfTopicEmbedder()


def _save_topic_source_documents(
    model_path: Path,
    documents: list[str],
    thinker_names: list[str],
) -> None:
    if len(documents) != len(thinker_names):
        raise ValueError(
            "Source documents and thinker names must have matching lengths."
        )
    documents_path = model_path / _TOPIC_SOURCE_DOCUMENTS_FILENAME
    temporary_path = documents_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            [
                {"name": name, "text": document}
                for name, document in zip(thinker_names, documents)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(documents_path)


def _load_topic_source_documents(documents_path: Path) -> list[dict[str, str]]:
    try:
        documents = json.loads(documents_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("The saved topic source documents are invalid.") from error
    if (
        not isinstance(documents, list)
        or not documents
        or any(
            not isinstance(document, dict)
            or not isinstance(document.get("name"), str)
            or not document["name"].strip()
            or not isinstance(document.get("text"), str)
            or not document["text"].strip()
            for document in documents
        )
    ):
        raise ValueError("The saved topic source documents are invalid.")
    return documents


def _approximate_thinker_year(name: str) -> int | None:
    match = re.fullmatch(
        r".*?\s+\((\d{1,4}(?: BC)?)(?:-(\d{1,4}(?: BC)?))?\)"
        r"(?:\s+\(active (\d{4})-(\d{4})\))?",
        name,
        re.IGNORECASE,
    )
    if not match:
        return None
    birth_start, birth_end, active_start, active_end = match.groups()
    birth_uses_bc = any(
        year and year.upper().endswith(" BC")
        for year in (birth_start, birth_end)
    )

    def parse_birth_year(year: str) -> int:
        return -int(year.removesuffix(" BC")) if birth_uses_bc else int(year)

    if active_start and active_end:
        approximate_year = (int(active_start) + int(active_end)) / 2
    elif birth_end:
        approximate_year = (
            parse_birth_year(birth_start) + parse_birth_year(birth_end)
        ) / 2
    else:
        approximate_year = parse_birth_year(birth_start)
    return math.floor(approximate_year + 0.5)


def _calculate_approximate_distributions(
    topic_model: Any,
    documents: list[dict[str, str]],
) -> list[dict[str, Any]]:
    topic_ids = sorted(
        topic_id for topic_id in topic_model.get_topics() if topic_id >= 0
    )
    if not documents or not topic_ids:
        return []
    topic_distributions, token_distributions = (
        topic_model.approximate_distribution(
            [document["text"] for document in documents],
            use_embedding_model=False,
            calculate_tokens=True,
        )
    )
    if (
        token_distributions is None
        or len(token_distributions) != len(documents)
        or len(topic_distributions) != len(documents)
    ):
        raise ValueError("The approximate topic distributions are incomplete.")
    distributions = []
    for document, probabilities, token_distribution in zip(
        documents, topic_distributions, token_distributions
    ):
        if len(probabilities) != len(topic_ids):
            raise ValueError(
                "The approximate distribution does not match the saved topics."
            )
        normalized_probabilities = [float(value) for value in probabilities]
        if any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in normalized_probabilities
        ):
            raise ValueError("The approximate topic distribution is invalid.")
        distributions.append(
            {
                "document": document,
                "topic_ids": topic_ids,
                "probabilities": normalized_probabilities,
                "token_distribution": token_distribution,
            }
        )
    return distributions


def _format_approximate_topic_distributions(
    topic_model: Any,
    distributions: list[dict[str, Any]],
    document_ids: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    document_ids = document_ids or {}
    topic_ids = distributions[0]["topic_ids"] if distributions else sorted(
        topic_id for topic_id in topic_model.get_topics() if topic_id >= 0
    )
    topics = []
    for index, topic_id in enumerate(topic_ids):
        terms = topic_model.get_topic(topic_id) or []
        name = " · ".join(term for term, _ in terms[:5] if term)
        if not name:
            name = f"Topic {topic_id}"
        thinkers = [
            {
                "name": distribution["document"]["name"],
                "document_id": document_ids.get(distribution["document"]["name"]),
                "probability": distribution["probabilities"][index],
            }
            for distribution in distributions
        ]
        topics.append(
            {
                "topic_id": topic_id,
                "name": name,
                "thinkers": thinkers,
            }
        )
    return topics


def _load_topic_search_embedder(vectorizer_path: Path) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        saved_vectorizer = json.loads(vectorizer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("The saved topic-search vectorizer is invalid.") from error

    if not isinstance(saved_vectorizer, dict):
        raise ValueError("The saved topic-search vectorizer is invalid.")
    if "embedding_model" in saved_vectorizer:
        return _sentence_transformer_backend(saved_vectorizer["embedding_model"])
    vocabulary = saved_vectorizer.get("vocabulary")
    idf = saved_vectorizer.get("idf")
    if not isinstance(vocabulary, dict) or not isinstance(idf, list):
        raise ValueError("The saved topic-search vectorizer is invalid.")
    if (
        not vocabulary
        or len(vocabulary) != len(idf)
        or any(
            not isinstance(term, str)
            or type(index) is not int
            or index < 0
            or index >= len(idf)
            for term, index in vocabulary.items()
        )
        or set(vocabulary.values()) != set(range(len(idf)))
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            for value in idf
        )
    ):
        raise ValueError("The saved topic-search vectorizer is invalid.")

    class TfidfTopicQueryEmbedder:
        def __init__(self) -> None:
            self.vocabulary = vocabulary
            self.idf = idf
            self.analyzer = TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
            ).build_analyzer()

        def embed_words(
            self,
            words: list[str],
            verbose: bool = False,
        ) -> Any:
            import numpy as np

            embeddings = np.zeros((len(words), len(self.idf)), dtype=float)
            for row, word in enumerate(words):
                for term in self.analyzer(word):
                    index = self.vocabulary.get(term)
                    if index is not None:
                        embeddings[row, index] += self.idf[index]
                norm = np.linalg.norm(embeddings[row])
                if norm:
                    embeddings[row] /= norm
            if not embeddings.any():
                raise ValueError(
                    "The topic query has no terms in the saved BERTopic vocabulary."
                )
            return embeddings

    return TfidfTopicQueryEmbedder()


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
            for field, field_type in spec.get("stored_fields", {}).items():
                existing_columns = {
                    row["name"]
                    for row in connection.execute(
                        f'PRAGMA table_info("{entity}")'
                    ).fetchall()
                }
                if field not in existing_columns:
                    connection.execute(
                        f'ALTER TABLE "{entity}" ADD COLUMN "{field}" {field_type}'
                    )
        connection.execute(
            'CREATE TABLE IF NOT EXISTS "concept_thinkers" ('
            "concept_id INTEGER NOT NULL, "
            "thinker_id INTEGER NOT NULL, "
            "PRIMARY KEY (concept_id, thinker_id))"
        )
        _migrate_generated_concept_relations(connection)


def _migrate_generated_concept_relations(
    connection: sqlite3.Connection,
) -> None:
    prefix = "BERTopic found this theme in the Wikipedia text of "
    legacy_concepts = connection.execute(
        'SELECT id, description FROM "concepts" '
        "WHERE description LIKE ?",
        (f"{prefix}%",),
    ).fetchall()
    if not legacy_concepts:
        return
    thinkers = connection.execute(
        'SELECT id, name FROM "thinkers"'
    ).fetchall()
    thinker_ids_by_name = {thinker["name"]: thinker["id"] for thinker in thinkers}
    for concept in legacy_concepts:
        legacy_text = concept["description"][len(prefix) :]
        contributor_text, separator, key_terms_text = legacy_text.partition(
            ". Key terms:"
        )
        if not separator:
            contributor_text = contributor_text.removesuffix(".")
        else:
            key_terms = [
                term.strip()
                for term in key_terms_text.removesuffix(".").split(",")
                if term.strip()
            ]
            if key_terms:
                connection.execute(
                    'UPDATE "concepts" SET name = ? WHERE id = ?',
                    (" · ".join(key_terms), concept["id"]),
                )
        for name in contributor_text.split(", "):
            thinker_id = thinker_ids_by_name.get(name)
            if thinker_id is not None:
                connection.execute(
                    'INSERT OR IGNORE INTO "concept_thinkers" '
                    "(concept_id, thinker_id) VALUES (?, ?)",
                    (concept["id"], thinker_id),
                )
        connection.execute(
            'UPDATE "concepts" SET description = ? WHERE id = ?',
            ("", concept["id"]),
        )


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _serialize_concept(
    row: sqlite3.Row, connection: sqlite3.Connection
) -> dict[str, Any]:
    concept = _serialize(row)
    thinkers = connection.execute(
        'SELECT thinkers.id, thinkers.name, thinkers.wiki_text_document_id, '
        'thinkers.military_thinking_document_id '
        'FROM "concept_thinkers" '
        'JOIN "thinkers" ON thinkers.id = concept_thinkers.thinker_id '
        'WHERE concept_thinkers.concept_id = ? '
        'ORDER BY thinkers.name COLLATE NOCASE',
        (concept["id"],),
    ).fetchall()
    concept["thinkers"] = [_serialize(thinker) for thinker in thinkers]
    assignments = concept.get("topic_assignments")
    concept["topic_assignments"] = (
        json.loads(assignments) if assignments else []
    )
    return concept


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


def _validate_remote_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as error:
        raise ValueError("The URL is invalid.") from error
    if (
        parsed.scheme != "https"
        or not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in allowed_hosts
        )
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        domains = " or ".join(allowed_hosts)
        raise ValueError(f"Enter an HTTPS URL from {domains}.")


def _wikipedia_article_key(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    return (
        (parsed.hostname or "").casefold(),
        unquote(parsed.path).rstrip("/").casefold(),
    )


class _AllowedHostsRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: tuple[str, ...]):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        _validate_remote_url(new_url, self.allowed_hosts)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _download_remote_content(
    url: str,
    *,
    allowed_hosts: tuple[str, ...],
    expected_content_type: str,
    timeout: int = REMOTE_REQUEST_TIMEOUT,
) -> tuple[bytes, str]:
    _validate_remote_url(url, allowed_hosts)
    page_request = Request(
        url,
        headers={
            "Accept": expected_content_type,
            "User-Agent": "BrunoConceptGenealogy/0.1 (local research tool)",
        },
    )
    opener = build_opener(_AllowedHostsRedirectHandler(allowed_hosts))
    with opener.open(page_request, timeout=timeout) as response:
        final_url = response.geturl()
        _validate_remote_url(final_url, allowed_hosts)
        if response.headers.get_content_type() != expected_content_type:
            raise ValueError(
                f"The URL did not return {expected_content_type} content."
            )
        content = response.read(MAX_DOWNLOAD_SIZE + 1)
    if len(content) > MAX_DOWNLOAD_SIZE:
        raise ValueError("The downloaded page is larger than the 10 MB limit.")
    return content, final_url


def _download_wikipedia_page(url: str) -> tuple[bytes, str]:
    return _download_remote_content(
        url,
        allowed_hosts=WIKIPEDIA_HOSTS,
        expected_content_type="text/html",
    )


def _download_wikipedia_text(url: str) -> str:
    _validate_remote_url(url, WIKIPEDIA_HOSTS)
    parsed_url = urlsplit(url)
    article_slug = unquote(parsed_url.path.rstrip("/").rsplit("/", 1)[-1])
    if not article_slug or article_slug in {".", ".."}:
        raise ValueError("The Wikipedia URL must point to an article.")
    api_params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": article_slug,
    }
    api_url = (
        f"https://{parsed_url.netloc}/w/api.php?{urlencode(api_params)}"
    )
    content, _ = _download_remote_content(
        api_url,
        allowed_hosts=WIKIPEDIA_HOSTS,
        expected_content_type="application/json",
    )
    try:
        result = json.loads(content)
        pages = result["query"]["pages"]
        page = next(iter(pages.values()))
        text = page["extract"]
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
    ) as error:
        raise ValueError("Wikipedia returned an invalid plain-text response.") from error
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Wikipedia did not return plain text for this article.")
    return text.strip()


def _dbpedia_data_url(wikipedia_url: str) -> str:
    _validate_remote_url(wikipedia_url, WIKIPEDIA_HOSTS)
    article_slug = unquote(urlsplit(wikipedia_url).path.rstrip("/").rsplit("/", 1)[-1])
    if not article_slug or article_slug in {".", ".."}:
        raise ValueError("The Wikipedia URL must point to an article.")
    return f"https://dbpedia.org/data/{quote(article_slug, safe='_()-')}.json"


def _download_dbpedia_data(url: str) -> tuple[bytes, str]:
    return _download_remote_content(
        url,
        allowed_hosts=DBPEDIA_HOSTS,
        expected_content_type="application/json",
        timeout=DBPEDIA_REQUEST_TIMEOUT,
    )


def _dbpedia_values(
    subject: dict[str, Any], predicate_suffixes: tuple[str, ...]
) -> list[dict[str, Any]]:
    for suffix in predicate_suffixes:
        for predicate, values in subject.items():
            if predicate.endswith(f"/{suffix}") or predicate.endswith(f"#{suffix}"):
                if isinstance(values, list):
                    literals = [
                        value
                        for value in values
                        if isinstance(value, dict)
                        and value.get("type") == "literal"
                        and isinstance(value.get("value"), (str, int))
                    ]
                    for value in literals:
                        if value.get("lang") == "en":
                            return literals
                    if literals:
                        return literals
    return []


def _dbpedia_literal(
    subject: dict[str, Any], predicate_suffixes: tuple[str, ...]
) -> str | None:
    values = _dbpedia_values(subject, predicate_suffixes)
    literals = [
        value
        for value in values
        if value.get("type") == "literal"
        and isinstance(value.get("value"), (str, int))
    ]
    for value in literals:
        if value.get("lang") == "en":
            return str(value["value"]).strip()
    for value in literals:
        if not value.get("lang"):
            return str(value["value"]).strip()
    return str(literals[0]["value"]).strip() if literals else None


def _dbpedia_subject(data: Any, wikipedia_url: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("DBpedia returned an invalid article.")
    article_slug = unquote(urlsplit(wikipedia_url).path.rstrip("/").rsplit("/", 1)[-1])
    expected_subject = f"http://dbpedia.org/resource/{article_slug}"
    subject = data.get(expected_subject)
    if subject is None:
        subject = next(
            (
                value
                for uri, value in data.items()
                if unquote(urlsplit(uri).path.rstrip("/").rsplit("/", 1)[-1]).casefold()
                == article_slug.casefold()
                and isinstance(value, dict)
            ),
            None,
        )
    if not isinstance(subject, dict):
        raise ValueError("DBpedia did not return data for this Wikipedia article.")
    return subject


def _dbpedia_resources(
    subject: dict[str, Any], predicate_suffixes: tuple[str, ...]
) -> list[dict[str, str]]:
    resources: list[dict[str, str]] = []
    for predicate, values in subject.items():
        if not any(
            predicate.endswith(f"/{suffix}") or predicate.endswith(f"#{suffix}")
            for suffix in predicate_suffixes
        ):
            continue
        if not isinstance(values, list):
            continue
        for value in values:
            if (
                not isinstance(value, dict)
                or value.get("type") != "uri"
                or not isinstance(value.get("value"), str)
            ):
                continue
            uri = value["value"]
            parsed_uri = urlsplit(uri)
            if parsed_uri.scheme not in {"http", "https"} or parsed_uri.hostname != "dbpedia.org":
                continue
            label = unquote(parsed_uri.path.rstrip("/").rsplit("/", 1)[-1])
            if label.startswith("Category:"):
                label = label.removeprefix("Category:")
            label = label.replace("_", " ")
            resources.append({"label": label, "url": uri})
    return sorted(resources, key=lambda item: item["label"].casefold())


def _dbpedia_display_value(
    subject: dict[str, Any], predicate_suffixes: tuple[str, ...]
) -> str | None:
    literal = _dbpedia_literal(subject, predicate_suffixes)
    if literal:
        return literal
    resources = _dbpedia_resources(subject, predicate_suffixes)
    return ", ".join(item["label"] for item in resources) or None


def _military_contributions_prompt(
    wikipedia_text: str,
    word_count: int = DEFAULT_RESPONSE_WORD_COUNT,
    prompt: str = DEFAULT_OLLAMA_PROMPT,
) -> str:
    return (
        f"{prompt.strip()}\n\n"
        f"Give a concise answer, approximately {word_count} words. "
        "Do not repeat biographical details unrelated "
        "to military strategy. Do not mention the thinker's name; refer to "
        "him only as 'he' or 'his'. Do not include named entities: omit names "
        "of people, places, organizations, battles, and countries, and omit "
        "years and specific dates. Describe strategic ideas in general terms. "
        "Make use of the provided Wikipedia article and your foundational "
        "knowledge on this thinker.\n\n"
        "Wikipedia article text:\n"
        f"{wikipedia_text}\n\n"
    )


def _document_view_data(
    document: dict[str, Any], documents: Any | None = None
) -> dict[str, Any]:
    source = document.get("source", "")
    source_labels = {
        "wikipedia-text": "Wikipedia article",
        "military-strategy": "Ollama analysis",
        "dbpedia": "DBpedia profile",
    }
    source_label = source_labels.get(source, "Source document")
    source_url = document.get("url", "")
    prompt = None
    if source == "military-strategy":
        prompt = document.get("prompt")
        if prompt is None and documents is not None:
            source_text = documents.find_one(
                {
                    "document_key": document.get("document_key"),
                    "source": "wikipedia-text",
                }
            )
            if source_text is not None:
                prompt = _military_contributions_prompt(
                    source_text["content"].decode("utf-8"),
                )
    if source == "dbpedia":
        data = json.loads(document["content"])
        dbpedia_path = urlsplit(source_url).path.rstrip("/")
        article_slug = dbpedia_path.rsplit("/", 1)[-1].removesuffix(".json")
        subject = _dbpedia_subject(
            data, f"https://en.wikipedia.org/wiki/{article_slug}"
        )
        title = "DBpedia data"
        facts = [
            {"label": "Birth date", "value": _dbpedia_year(_dbpedia_literal(subject, ("birthDate", "dateOfBirth")))},
            {"label": "Death date", "value": _dbpedia_year(_dbpedia_literal(subject, ("deathDate", "dateOfDeath")))},
            {"label": "Occupation", "value": _dbpedia_display_value(subject, ("occupation",))},
            {"label": "Nationality", "value": _dbpedia_display_value(subject, ("nationality",))},
            {"label": "Military branch", "value": _dbpedia_display_value(subject, ("militaryBranch",))},
        ]
        taxonomies = [
            {
                "heading": "DBpedia classes",
                "items": _dbpedia_resources(subject, ("type",)),
            },
            {
                "heading": "Wikipedia categories",
                "items": _dbpedia_resources(subject, ("subject",)),
            },
        ]
        return {
            "title": title,
            "source_label": source_label,
            "source_url": source_url,
            "facts": [fact for fact in facts if fact["value"]],
            "taxonomies": taxonomies,
            "paragraphs": [],
            "prompt": None,
        }

    content = document["content"]
    text = content.decode("utf-8")
    # Count words in the document text for Ollama analyses and expose to the template
    # A word is a run of word characters; this is a simple heuristic but effective
    # for displaying an approximate word count.
    word_matches = re.findall(r"\b\w+\b", text)
    word_count = len(word_matches)
    return {
        "title": source_labels.get(source, "Saved source"),
        "source_label": source_label,
        "source_url": source_url,
        "facts": [],
        "taxonomies": [],
        "paragraphs": [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()],
        "prompt": prompt,
        "word_count": word_count,
        "editable_text": text if source == "wikipedia-text" else None,
    }


def _thinker_name_from_dbpedia(
    dbpedia_content: bytes,
    wikipedia_url: str,
    wikipedia_content: bytes | None = None,
) -> str:
    try:
        data = json.loads(dbpedia_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DBpedia returned invalid JSON.") from error

    subject = _dbpedia_subject(data, wikipedia_url)
    person_name = _dbpedia_literal(
        subject,
        (
            "name",
            "label",
        ),
    )
    if not person_name:
        raise ValueError("DBpedia did not provide a name for this thinker.")

    birth_date = _dbpedia_literal(subject, ("birthDate", "dateOfBirth"))
    death_date = _dbpedia_literal(subject, ("deathDate", "dateOfDeath"))
    birth_year = _dbpedia_year(birth_date, wikipedia_content)
    death_year = _dbpedia_year(death_date, wikipedia_content)
    years: str | None = None
    if birth_year and death_year:
        if birth_year.endswith(" BC") and death_year.endswith(" BC"):
            years = f"{birth_year.removesuffix(' BC')}-{death_year}"
        else:
            years = f"{birth_year}-{death_year}"
    elif birth_year:
        years = birth_year
    elif death_year:
        years = death_year

    active_start = _dbpedia_literal(
        subject,
        ("activeYearsStartYear", "serviceStartYear", "careerStartYear"),
    )
    active_end = _dbpedia_literal(
        subject,
        ("activeYearsEndYear", "serviceEndYear", "careerEndYear"),
    )
    active_period: str | None = None
    if active_start and active_end:
        active_period = f"active {active_start}-{active_end}"

    if years and active_period:
        return f"{person_name} ({years}) ({active_period})"
    if years:
        return f"{person_name} ({years})"
    if active_period:
        return f"{person_name} ({active_period})"
    return person_name


def _dbpedia_year(
    value: str | None, wikipedia_content: bytes | None = None
) -> str | None:
    if not value:
        return None
    match = re.fullmatch(
        r"\s*(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+)?"
        r"(-?\d{1,4})(?:-\d{2}(?:-\d{2})?)?\s*(BC|BCE)?\s*",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    year = abs(int(match.group(1)))
    is_bc = (
        int(match.group(1)) < 0
        or bool(match.group(2))
        or _wikipedia_confirms_bc_year(year, wikipedia_content)
    )
    return f"{year} BC" if is_bc else str(year)


def _wikipedia_confirms_bc_year(
    year: int, wikipedia_content: bytes | None
) -> bool:
    if wikipedia_content is None:
        return False
    page_text = html.unescape(
        re.sub(
            r"<[^>]*>",
            " ",
            wikipedia_content.decode("utf-8", errors="replace"),
        )
    )
    era = r"(?:B\.?\s*C\.?|BCE)"
    return bool(
        re.search(
            rf"(?:\b{year}\s*{era}\b|\b{era}\s*{year}\b)",
            page_text,
            re.IGNORECASE,
        )
    )


def _extract_military_contributions(
    wikipedia_text: str,
    word_count: int = DEFAULT_RESPONSE_WORD_COUNT,
    prompt: str = DEFAULT_OLLAMA_PROMPT,
) -> str:
    model = os.environ.get("BRUNO_LLM_MODEL", "qwen2.5:3b")
    llm_url = os.environ.get(
        "BRUNO_LLM_URL", "http://localhost:11434/api/generate"
    )
    model_request = Request(
        llm_url,
        data=json.dumps(
            {
                "model": model,
                "stream": False,
                "prompt": _military_contributions_prompt(
                    wikipedia_text,
                    word_count,
                    prompt,
                ),
                "options": {
                    "num_ctx": 8192,
                    "num_predict": max(300, math.ceil(word_count * 1.5)),
                },
                "keep_alive": "10m",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(model_request, timeout=LLM_REQUEST_TIMEOUT) as response:
        model_result = json.loads(response.read())
    contributions = model_result["response"]
    if not isinstance(contributions, str) or not contributions.strip():
        raise ValueError("The model returned an empty military contributions response.")
    return contributions.strip()


def _language_model_connected() -> bool:
    model = os.environ.get("BRUNO_LLM_MODEL", "qwen2.5:3b")
    llm_url = os.environ.get(
        "BRUNO_LLM_URL", "http://localhost:11434/api/generate"
    )
    parsed_url = urlsplit(llm_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return False
    tags_url = f"{parsed_url.scheme}://{parsed_url.netloc}/api/tags"
    try:
        with urlopen(tags_url, timeout=3) as response:
            model_result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return False
    models = model_result.get("models") if isinstance(model_result, dict) else None
    if not isinstance(models, list):
        return False
    available_models = {
        item.get("name")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    expected_models = {model}
    if ":" not in model:
        expected_models.add(f"{model}:latest")
    return bool(expected_models & available_models)


def _store_document(
    collection: Any,
    document_key: str,
    source: str,
    url: str,
    content: bytes,
    content_type: str,
    prompt: str | None = None,
) -> str:
    document = {
        "document_key": document_key,
        "source": source,
        "url": url,
        "content_type": content_type,
        "content": content,
    }
    if prompt is not None:
        document["prompt"] = prompt
    result = collection.insert_one(document)
    return str(result.inserted_id)


def _register_entity_routes(
    app: Flask, entity: str, spec: EntitySpec
) -> None:
    table = entity

    def collection():
        with _database(app.config["DATABASE"]) as connection:
            rows = connection.execute(
                f'SELECT * FROM "{table}" ORDER BY id DESC'
            ).fetchall()
            entries = [
                _serialize_concept(row, connection)
                if entity == "concepts"
                else _serialize(row)
                for row in rows
            ]
        return jsonify(entries)

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
            entry = (
                _serialize_concept(row, connection)
                if entity == "concepts"
                else _serialize(row)
            )
        return jsonify(entry), 201

    def item(item_id: int):
        with _database(app.config["DATABASE"]) as connection:
            row = connection.execute(
                f'SELECT * FROM "{table}" WHERE id = ?', (item_id,)
            ).fetchone()
            if row is not None:
                entry = (
                    _serialize_concept(row, connection)
                    if entity == "concepts"
                    else _serialize(row)
                )
        if row is None:
            return jsonify({"error": f'{spec["label"]} not found.'}), 404
        return jsonify(entry)

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
            entry = (
                _serialize_concept(row, connection)
                if entity == "concepts"
                else _serialize(row)
            )
        return jsonify(entry)

    def delete(item_id: int):
        if entity == "thinkers":
            with _database(app.config["DATABASE"]) as connection:
                existing = connection.execute(
                    'SELECT wiki_document_key FROM "thinkers" WHERE id = ?',
                    (item_id,),
                ).fetchone()
            if existing is None:
                return jsonify({"error": f'{spec["label"]} not found.'}), 404
            if existing["wiki_document_key"]:
                try:
                    app.config["MONGO_DOCUMENTS"].delete_many(
                        {"document_key": existing["wiki_document_key"]}
                    )
                except PyMongoError as error:
                    return jsonify({"error": f"MongoDB is unavailable: {error}"}), 503
        with _database(app.config["DATABASE"]) as connection:
            if entity == "concepts":
                connection.execute(
                    'DELETE FROM "concept_thinkers" WHERE concept_id = ?',
                    (item_id,),
                )
            elif entity == "thinkers":
                connection.execute(
                    'DELETE FROM "concept_thinkers" WHERE thinker_id = ?',
                    (item_id,),
                )
            cursor = connection.execute(
                f'DELETE FROM "{table}" WHERE id = ?', (item_id,)
            )
        if cursor.rowcount == 0:
            return jsonify({"error": f'{spec["label"]} not found.'}), 404
        return "", 204

    def process_thinker(item_id: int | None = None):
        raw_payload = request.get_json(silent=True)
        word_count = DEFAULT_RESPONSE_WORD_COUNT
        analysis_prompt = DEFAULT_OLLAMA_PROMPT
        if isinstance(raw_payload, dict) and "word_count" in raw_payload:
            requested_word_count = raw_payload["word_count"]
            if (
                isinstance(requested_word_count, bool)
                or not isinstance(requested_word_count, int)
                or not MIN_RESPONSE_WORD_COUNT
                <= requested_word_count
                <= MAX_RESPONSE_WORD_COUNT
            ):
                return jsonify(
                    {
                        "error": (
                            "The analysis word count must be an integer "
                            f"between {MIN_RESPONSE_WORD_COUNT} and "
                            f"{MAX_RESPONSE_WORD_COUNT}."
                        )
                    }
                ), 400
            word_count = requested_word_count
        if isinstance(raw_payload, dict) and "prompt" in raw_payload:
            requested_prompt = raw_payload["prompt"]
            if (
                not isinstance(requested_prompt, str)
                or not requested_prompt.strip()
                or len(requested_prompt) > MAX_OLLAMA_PROMPT_LENGTH
            ):
                return jsonify(
                    {
                        "error": (
                            "The Ollama prompt must be non-empty and no longer "
                            f"than {MAX_OLLAMA_PROMPT_LENGTH} characters."
                        )
                    }
                ), 400
            analysis_prompt = requested_prompt.strip()
        thinker_payload = (
            {
                key: value
                for key, value in raw_payload.items()
            }
            if isinstance(raw_payload, dict)
            else raw_payload
        )
        # Remove control keys not part of the thinker schema before validation.
        if isinstance(thinker_payload, dict):
            # work on a shallow copy so we don't mutate the original raw_payload
            thinker_payload = dict(thinker_payload)
            thinker_payload.pop("prompt", None)
            thinker_payload.pop("word_count", None)
        payload, error = _validate_payload(
            entity, thinker_payload, partial=True
        )
        if error:
            return jsonify({"error": error}), 400
        assert payload is not None
        wikipedia_url = payload.get("biography")
        if not wikipedia_url:
            return jsonify({"error": "The Wikipedia URL is required."}), 400
        try:
            _validate_remote_url(wikipedia_url, WIKIPEDIA_HOSTS)
        except ValueError as url_error:
            return jsonify({"error": str(url_error)}), 400

        if item_id is not None:
            with _database(app.config["DATABASE"]) as connection:
                existing = connection.execute(
                    'SELECT * FROM "thinkers" WHERE id = ?', (item_id,)
                ).fetchone()
            if existing is None:
                return jsonify({"error": "Thinker not found."}), 404
        else:
            existing = None

        try:
            wiki_content, final_wikipedia_url = _download_wikipedia_page(wikipedia_url)
        except ValueError as download_error:
            return jsonify({"error": str(download_error)}), 400
        except (HTTPError, URLError, TimeoutError, OSError) as download_error:
            return jsonify(
                {"error": f"Could not download the Wikipedia page: {download_error}"}
            ), 502

        try:
            wikipedia_text = _download_wikipedia_text(final_wikipedia_url)
        except ValueError as download_error:
            return jsonify(
                {"error": f"Could not retrieve Wikipedia plain text: {download_error}"}
            ), 502
        except (HTTPError, URLError, TimeoutError, OSError) as download_error:
            return jsonify(
                {"error": f"Could not retrieve Wikipedia plain text: {download_error}"}
            ), 502

        try:
            dbpedia_url = _dbpedia_data_url(final_wikipedia_url)
            dbpedia_content, final_dbpedia_url = _download_dbpedia_data(dbpedia_url)
        except ValueError as download_error:
            return jsonify({"error": str(download_error)}), 400
        except (HTTPError, URLError, TimeoutError, OSError) as download_error:
            return jsonify(
                {"error": f"Could not download DBpedia data: {download_error}"}
            ), 502

        try:
            name = _thinker_name_from_dbpedia(
                dbpedia_content, final_wikipedia_url, wiki_content
            )
        except ValueError as parse_error:
            return jsonify(
                {"error": f"Could not identify the thinker in DBpedia: {parse_error}"}
            ), 502

        military_prompt = _military_contributions_prompt(
            wikipedia_text,
            word_count,
            analysis_prompt,
        )
        try:
            military_contributions = _extract_military_contributions(
                wikipedia_text,
                word_count,
                analysis_prompt,
            )
        except (HTTPError, URLError, TimeoutError, OSError) as model_error:
            return jsonify(
                {"error": f"The language model is unavailable: {model_error}"}
            ), 503
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as model_error:
            return jsonify(
                {"error": f"The language model returned an invalid response: {model_error}"}
            ), 502

        with _database(app.config["DATABASE"]) as connection:
            matching_thinkers = [
                row
                for row in connection.execute(
                    'SELECT * FROM "thinkers" WHERE biography IS NOT NULL ORDER BY id'
                ).fetchall()
                if _wikipedia_article_key(row["biography"])
                == _wikipedia_article_key(final_wikipedia_url)
            ]

        if item_id is None and matching_thinkers:
            existing = matching_thinkers[0]
            item_id = existing["id"]
        duplicate_thinkers = [
            row for row in matching_thinkers if row["id"] != item_id
        ]
        old_document_keys = {
            row["wiki_document_key"]
            for row in [existing, *duplicate_thinkers]
            if row is not None and row["wiki_document_key"]
        }

        document_key = uuid.uuid4().hex
        inserted_ids: list[str] = []
        documents = app.config["MONGO_DOCUMENTS"]
        try:
            wiki_document_id = _store_document(
                documents,
                document_key,
                "wikipedia",
                final_wikipedia_url,
                wiki_content,
                "text/html",
            )
            inserted_ids.append(wiki_document_id)
            wiki_text_document_id = _store_document(
                documents,
                document_key,
                "wikipedia-text",
                final_wikipedia_url,
                wikipedia_text.encode("utf-8"),
                "text/plain; charset=utf-8",
            )
            inserted_ids.append(wiki_text_document_id)
            dbpedia_document_id = _store_document(
                documents,
                document_key,
                "dbpedia",
                final_dbpedia_url,
                dbpedia_content,
                "application/json",
            )
            inserted_ids.append(dbpedia_document_id)
            military_thinking_document_id = _store_document(
                documents,
                document_key,
                "military-strategy",
                final_wikipedia_url,
                military_contributions.encode("utf-8"),
                "text/plain; charset=utf-8",
                prompt=military_prompt,
            )
            inserted_ids.append(military_thinking_document_id)
        except PyMongoError as store_error:
            if inserted_ids:
                try:
                    documents.delete_many(
                        {"_id": {"$in": [ObjectId(value) for value in inserted_ids]}}
                    )
                except PyMongoError as cleanup_error:
                    return jsonify(
                        {
                            "error": (
                                f"Could not store source documents: {store_error}; "
                                f"partial-document cleanup also failed: {cleanup_error}"
                            )
                        }
                    ), 503
            return jsonify({"error": f"MongoDB is unavailable: {store_error}"}), 503

        try:
            with _database(app.config["DATABASE"]) as connection:
                if item_id is None:
                    cursor = connection.execute(
                        'INSERT INTO "thinkers" '
                        "(name, biography, wiki_document_key, wiki_document_id, "
                        "wiki_text_document_id, dbpedia_document_id, "
                        "military_thinking_document_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            name,
                            final_wikipedia_url,
                            document_key,
                            wiki_document_id,
                            wiki_text_document_id,
                            dbpedia_document_id,
                            military_thinking_document_id,
                        ),
                    )
                    row = connection.execute(
                        'SELECT * FROM "thinkers" WHERE id = ?', (cursor.lastrowid,)
                    ).fetchone()
                    status_code = 201
                else:
                    connection.execute(
                        'UPDATE "thinkers" SET name = ?, biography = ?, '
                        "wiki_document_key = ?, wiki_document_id = ?, "
                        "wiki_text_document_id = ?, dbpedia_document_id = ?, "
                        "military_thinking_document_id = ? "
                        "WHERE id = ?",
                        (
                            name,
                            final_wikipedia_url,
                            document_key,
                            wiki_document_id,
                            wiki_text_document_id,
                            dbpedia_document_id,
                            military_thinking_document_id,
                            item_id,
                        ),
                    )
                    row = connection.execute(
                        'SELECT * FROM "thinkers" WHERE id = ?', (item_id,)
                    ).fetchone()
                    status_code = 200
                    if duplicate_thinkers:
                        duplicate_ids = [duplicate["id"] for duplicate in duplicate_thinkers]
                        placeholders = ", ".join("?" for _ in duplicate_ids)
                        connection.execute(
                            f'DELETE FROM "thinkers" WHERE id IN ({placeholders})',
                            duplicate_ids,
                        )
        except sqlite3.Error as database_error:
            try:
                documents.delete_many({"document_key": document_key})
            except PyMongoError as cleanup_error:
                return jsonify(
                    {
                        "error": (
                            f"Could not save thinker metadata: {database_error}; "
                            f"MongoDB cleanup also failed: {cleanup_error}"
                        )
                    }
                ), 500
            return jsonify(
                {"error": f"Could not save thinker metadata: {database_error}"}
            ), 500

        old_document_keys.discard(document_key)
        for old_document_key in old_document_keys:
            try:
                documents.delete_many({"document_key": old_document_key})
            except PyMongoError as cleanup_error:
                return jsonify(
                    {
                        "error": (
                            "Thinker was updated, but old source documents could not "
                            f"be removed: {cleanup_error}"
                        )
                    }
                ), 503
        return jsonify(_serialize(row)), status_code

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
    if entity == "thinkers":
        app.add_url_rule(
            "/api/thinkers/process",
            view_func=process_thinker,
            methods=["POST"],
        )
        app.add_url_rule(
            "/api/thinkers/<int:item_id>/process",
            view_func=process_thinker,
            methods=["PATCH"],
        )


def main() -> None:
    """Run the web application."""
    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )


if __name__ == "__main__":
    main()
