const collections = {
  concepts: {
    label: "Topic",
    plural: "Topics",
    detailField: "description",
    detailLabel: "Description",
    placeholder: "What does this topic mean?",
  },
  thinkers: {
    label: "Thinker",
    plural: "Thinkers",
    detailField: "biography",
    detailLabel: "Wikipedia",
    placeholder: "https://en.wikipedia.org/wiki/Carl_von_Clausewitz",
  },
};

const form = document.querySelector("#entry-form");
const nameInput = document.querySelector("#entry-name");
const nameLabel = document.querySelector("#name-label");
const detailsInput = document.querySelector("#entry-details");
const wikipediaInput = document.querySelector("#entry-wikipedia");
const heading = document.querySelector("#form-heading");
const formEyebrow = document.querySelector("#form-eyebrow");
const responseWordCountLabel = document.querySelector("#response-word-count-label");
const responseWordCountInput = document.querySelector("#response-word-count");
const responseWordCountHelp = document.querySelector("#response-word-count-help");
const ollamaPromptInput = document.querySelector("#ollama-prompt");
const ollamaPromptLabel = document.querySelector("#ollama-prompt-label");
const ollamaPromptHelp = document.querySelector("#ollama-prompt-help");
const wikipediaSuggestions = document.querySelector("#wikipedia-suggestions");
const topicSourceLabel = document.querySelector("#topic-source-label");
const topicSourceSelect = document.querySelector("#topic-source");
const topicCountLabel = document.querySelector("#topic-count-label");
const topicCountInput = document.querySelector("#topic-count");
const topicCountHelp = document.querySelector("#topic-count-help");
const topicModelOptions = document.querySelector("#topic-model-options");
const topicMinDfInput = document.querySelector("#topic-min-df");
const topicMaxWordsInput = document.querySelector("#topic-max-words");
const topicTopNWordsInput = document.querySelector("#topic-top-n-words");
const topicDiversityInput = document.querySelector("#topic-diversity");
const detailsLabel = document.querySelector("#details-label");
const saveButton = document.querySelector("#save-button");
const cancelButton = document.querySelector("#cancel-button");
const list = document.querySelector("#entry-list");
const formError = document.querySelector("#form-error");
const formStatus = document.querySelector("#form-status");
const listError = document.querySelector("#list-error");
const topicSearchBox = document.querySelector("#topic-search-box");
const topicSearchForm = document.querySelector("#topic-search-form");
const topicSearchConceptInput = document.querySelector("#topic-search-concept");
const topicSearchButton = document.querySelector("#topic-search-button");
const topicSearchError = document.querySelector("#topic-search-error");
const topicSearchResults = document.querySelector("#topic-search-results");
const topicInformation = document.querySelector("#topic-information");
const thinkerInformation = document.querySelector("#thinker-information");
const apiStatus = document.querySelector("#api-status");
const languageModelStatus = document.querySelector("#language-model-status");
const workspace = document.querySelector(".workspace");
const visualizationsPanel = document.querySelector("#visualizations");
const topicsVisualization = document.querySelector("#topics-visualization");
const visualizationError = document.querySelector("#visualization-error");
const visualizationLoading = document.querySelector("#visualization-loading");
const visualizationOptions = [...document.querySelectorAll(".visualization-option")];
const visualizationCount = document.querySelector("#visualization-count");
const thinkerSearch = document.querySelector("#thinker-search");
const distributionThinkerSearch = document.querySelector("#distribution-thinker-search");
const distributionStatus = document.querySelector("#topic-distribution-status");
const distributionRetry = document.querySelector("#topic-distribution-retry");
const distributionBubbles = document.querySelector("#topic-distribution-bubbles");
const distributionPanel = document.querySelector("#topic-distribution-panel");
const distributionPanelAssignments = document.querySelector("#topic-distribution-panel-assignments");

let activeCollection = "thinkers";
let editingId = null;
let conceptCount = 0;
let visualizationRequestId = 0;
let distributionRequestId = 0;
let distributionLoadPromise = null;
let distributionsLoaded = false;
let topicDistributions = [];

const visualizationCountText = String(visualizationOptions.length);
visualizationCount.textContent = visualizationCountText;
visualizationCount.setAttribute(
  "aria-label",
  `${visualizationCountText} available visualizations`,
);

window.addEventListener("message", (event) => {
  if (
    event.source !== topicsVisualization.contentWindow
    || !event.data
    || event.data.type !== "visualization-height"
    || !Number.isFinite(event.data.height)
    || event.data.height < 1
  ) {
    return;
  }
  topicsVisualization.style.height = `${Math.ceil(event.data.height)}px`;
});

topicsVisualization.addEventListener("load", () => {
  topicsVisualization.contentWindow?.postMessage(
    {
      type: "thinker-search",
      query: distributionThinkerSearch.value,
    },
    "*",
  );
});

function searchMatches(text, query) {
  return text.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());
}

function updateTopicDistributionBubbles() {
  for (const [index, distribution] of topicDistributions.entries()) {
    const bubble = distributionBubbles.children[index];
    bubble.setAttribute(
      "aria-label",
      `Topic ${distribution.topic_id}: ${distribution.name}`,
    );
  }
}

function _createAssignmentsList(thinkers) {
  // Returns a DOM node with the same structure used in the topics tab:
  // <div class="topic-assignments"><ul class="concept-thinkers">...</ul></div>
  const assignments = document.createElement("div");
  assignments.className = "topic-assignments";
  const assignmentList = document.createElement("ul");
  assignmentList.className = "concept-thinkers";

  const ordered = [...(thinkers ?? [])].sort((left, right) => {
    const lp = Number.isFinite(left.probability) ? left.probability : -1;
    const rp = Number.isFinite(right.probability) ? right.probability : -1;
    return rp - lp;
  });

  for (const thinker of ordered) {
    const item = document.createElement("li");
    // Prefer showing the name without year ranges — reuse thinkerTimeline helper
    const displayName = thinkerTimeline?.(thinker.name)?.name ?? thinker.name;
    if (thinker.document_id) {
      const link = document.createElement("a");
      link.href = `/documents/${encodeURIComponent(thinker.document_id)}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = displayName;
      item.append(link);
    } else {
      item.textContent = displayName;
    }
    if (Number.isFinite(thinker.probability)) {
      item.append(
        document.createTextNode(
          ` (${Math.round(thinker.probability * 100)}%)`,
        ),
      );
    }
    assignmentList.append(item);
  }

  if (assignmentList.childElementCount > 0) {
    assignments.append(assignmentList);
  } else {
    const emptyAssignments = document.createElement("p");
    emptyAssignments.className = "topic-assignments-empty";
    emptyAssignments.textContent = "No topic assignments.";
    assignments.append(emptyAssignments);
  }
  return assignments;
}

function showTopicDistributionDetails(topic, bubble) {
  const isSelected = bubble.classList.contains("selected");
  if (isSelected) {
    hideTopicDistributionDetails();
    return;
  }

  const query = distributionThinkerSearch.value;
  const filteredThinkers = (topic.thinkers || [])
    .filter((thinker) => searchMatches(thinker.name, query))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, 5)
    .map((t) => ({ ...t }));

  distributionPanelAssignments.replaceChildren();

  const assignments = _createAssignmentsList(filteredThinkers);
  distributionPanelAssignments.append(assignments);
  distributionPanel.hidden = false;

  [...distributionBubbles.children].forEach((button) => {
    const selected = button === bubble;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-expanded", String(selected));
  });
}

function hideTopicDistributionDetails() {
  distributionPanel.hidden = true;
  distributionPanelAssignments.replaceChildren();
  [...distributionBubbles.children].forEach((button) => {
    button.classList.remove("selected");
    button.setAttribute("aria-expanded", "false");
  });
}

async function loadTopicDistributions() {
  if (distributionsLoaded) return;
  if (distributionLoadPromise) return distributionLoadPromise;
  // Defensive guards: if essential DOM nodes are missing, log and skip to avoid
  // throwing a TypeError that prevents further UI updates.
  if (!distributionBubbles || !distributionStatus || !distributionPanel || !distributionRetry) {
    console.error("Topic distribution UI elements are missing", {
      distributionBubbles: !!distributionBubbles,
      distributionPanel: !!distributionPanel,
      distributionStatus: !!distributionStatus,
      distributionRetry: !!distributionRetry,
    });
    return;
  }

  const requestId = ++distributionRequestId;
  distributionBubbles.replaceChildren();
  topicDistributions = [];
  distributionRetry.hidden = true;
  distributionStatus.classList.remove("error");
  distributionStatus.textContent = "Calculating approximate distributions…";
  distributionLoadPromise = (async () => {
    try {
      const response = await requestJson(
        "/api/visualizations/approximate-distributions",
        { cache: "no-store" },
      );
      if (requestId !== distributionRequestId) return;
      topicDistributions = response.topics;
      for (const topic of topicDistributions) {
        const bubble = document.createElement("button");
        bubble.className = "topic-distribution-bubble";
        bubble.type = "button";
        const topicNumber = document.createElement("span");
        topicNumber.className = "topic-distribution-bubble-number";
        topicNumber.textContent = String(topic.topic_id);
        const topicTerms = document.createElement("span");
        topicTerms.className = "topic-distribution-bubble-name";
        topicTerms.textContent = topic.name;
        bubble.append(topicNumber, topicTerms);
        bubble.setAttribute(
          "aria-label",
          `Topic ${topic.topic_id}: ${topic.name}`,
        );
        bubble.setAttribute("aria-controls", "topic-distribution-panel");
        bubble.setAttribute("aria-expanded", "false");
        bubble.addEventListener("click", () =>
          showTopicDistributionDetails(topic, bubble),
        );
        distributionBubbles.append(bubble);
      }
      distributionsLoaded = true;
      distributionStatus.textContent = topicDistributions.length
        ? ""
        : "No topics are available.";
      updateTopicDistributionBubbles();
    } catch (error) {
      console.error("Failed to load topic distributions", error);
      if (requestId === distributionRequestId) {
        distributionStatus.textContent = error.message || String(error);
        distributionStatus.classList.add("error");
        distributionRetry.hidden = false;
      }
    }
  })();
  try {
    await distributionLoadPromise;
  } finally {
    if (requestId === distributionRequestId) {
      distributionLoadPromise = null;
    }
  }
}

async function loadVisualization() {
  const requestId = ++visualizationRequestId;
  const selectedOption = visualizationOptions.find(
    (option) => option.getAttribute("aria-pressed") === "true",
  );
  if (!selectedOption) {
    throw new Error("Choose a visualization to display.");
  }
  const visualization = selectedOption.dataset.visualization;
  distributionThinkerSearch.hidden =
    visualization !== "approximate-distribution";
    if (visualization !== "approximate-distribution") {
      hideTopicDistributionDetails();
    } else {
      // ensure panel is hidden when switching in but no hover yet
      distributionPanel.hidden = true;
    }
  showError(visualizationError);
  visualizationLoading.textContent = visualization === "approximate-distribution"
    ? "Calculating approximate distributions for the saved source texts…"
    : visualization === "topics-over-time"
      ? "Loading topics over time using the thinkers' approximate years…"
      : visualization === "topics-per-class"
        ? "Calculating topics per thinker from the saved source texts…"
        : visualization === "hierarchy"
          ? "Loading the hierarchical topic visualization…"
          : visualization === "hierarchical-documents"
            ? "Calculating the hierarchical thinkers and topics from the saved source texts… This can take several minutes."
            : visualization === "heatmap"
            ? "Loading the topic similarity heatmap…"
            : "Loading…";
  visualizationLoading.hidden = false;
  topicsVisualization.srcdoc = "";
  topicsVisualization.style.height = "0px";
  try {
    const response = await fetch(`/visualizations/${visualization}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(
        `Could not load ${visualization.replaceAll("-", " ")} visualization (HTTP ${response.status}).`,
      );
    }
    const html = await response.text();
    if (requestId === visualizationRequestId) {
      topicsVisualization.srcdoc = html;
    }
  } catch (error) {
    if (requestId === visualizationRequestId) {
      showError(visualizationError, error.message);
    }
  } finally {
    if (requestId === visualizationRequestId) {
      visualizationLoading.hidden = true;
    }
  }
}

function showError(element, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function setConnectionStatus(element, connected, label) {
  element.querySelector(".status-label").textContent = label;
  element.classList.toggle("unavailable", !connected);
}

function setEditing(entry = null) {
  editingId = entry ? entry.id : null;
  const config = collections[activeCollection];
  // Guard element access in case the DOM is not exactly as expected (prevents TypeError)
  if (wikipediaSuggestions) wikipediaSuggestions.hidden = activeCollection !== "thinkers";
  if (responseWordCountLabel) responseWordCountLabel.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (responseWordCountInput) responseWordCountInput.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (responseWordCountHelp) responseWordCountHelp.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (ollamaPromptLabel) ollamaPromptLabel.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (ollamaPromptInput) ollamaPromptInput.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (ollamaPromptHelp) ollamaPromptHelp.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (thinkerSearch) thinkerSearch.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (topicSourceLabel) topicSourceLabel.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (topicSourceSelect) topicSourceSelect.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (topicCountLabel) topicCountLabel.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (topicCountInput) topicCountInput.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (topicCountHelp) topicCountHelp.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (topicModelOptions) topicModelOptions.hidden = activeCollection !== "concepts" || Boolean(entry);
  const topicOptionsDisabled = activeCollection !== "concepts" || Boolean(entry);
  if (topicMinDfInput) topicMinDfInput.disabled = topicOptionsDisabled;
  if (topicMaxWordsInput) topicMaxWordsInput.disabled = topicOptionsDisabled;
  if (topicTopNWordsInput) topicTopNWordsInput.disabled = topicOptionsDisabled;
  if (topicDiversityInput) topicDiversityInput.disabled = topicOptionsDisabled;
  if (topicSearchBox) topicSearchBox.hidden = activeCollection !== "concepts" || Boolean(entry) || conceptCount < 1;
  if (topicInformation) topicInformation.hidden = activeCollection !== "concepts" || Boolean(entry);
  if (thinkerInformation) thinkerInformation.hidden = activeCollection !== "thinkers" || Boolean(entry);
  if (formEyebrow) formEyebrow.hidden = activeCollection === "concepts" && !entry;
  if (formEyebrow) formEyebrow.textContent = entry
    ? "EDIT ENTRY"
    : activeCollection === "concepts" ? "" : "NEW ENTRY";
  if (heading) heading.textContent = entry
    ? `Edit ${config.label.toLowerCase()}`
    : activeCollection === "concepts"
      ? "Generate topics"
      : `Add a ${config.label.toLowerCase()}`;
  const actionLabel = activeCollection === "concepts"
    ? "topics"
    : config.label.toLowerCase();
  if (saveButton) saveButton.innerHTML = entry
    ? 'Save changes <span aria-hidden="true">→</span>'
    : `${activeCollection === "thinkers" ? "Process" : "Generate"} ${actionLabel} <span aria-hidden="true">→</span>`;
  if (cancelButton) cancelButton.hidden = !entry;
  if (nameLabel) nameLabel.hidden = activeCollection !== "concepts" || !entry;
  if (nameInput) nameInput.hidden = activeCollection !== "concepts" || !entry;
  if (nameInput) nameInput.required = activeCollection === "concepts" && Boolean(entry);
  if (detailsLabel) detailsLabel.hidden = activeCollection !== "thinkers" && !entry;
  if (detailsLabel) detailsLabel.textContent = config.detailLabel;
  if (detailsLabel) detailsLabel.htmlFor = activeCollection === "thinkers" ? "entry-wikipedia" : "entry-details";
  if (detailsInput) detailsInput.hidden = activeCollection !== "concepts" || !entry;
  if (detailsInput) detailsInput.placeholder = config.placeholder;
  if (wikipediaInput) wikipediaInput.hidden = activeCollection !== "thinkers";
  if (nameInput) nameInput.value = entry?.name ?? "";
  if (detailsInput) detailsInput.value = "";
  if (wikipediaInput) wikipediaInput.value = "";
  if (responseWordCountInput) responseWordCountInput.value = "500";
  if (ollamaPromptInput) {
    // Only set the default prompt when the field is currently empty or when editing an entry.
    // Do not overwrite a user-provided custom prompt so reprocess uses the input field's value.
    if (entry || ollamaPromptInput.value.trim() === "") {
      ollamaPromptInput.value = "What are the thinker's contributions to military strategic thought?";
    }
  }
  if (topicCountInput) topicCountInput.value = "";
  const detailInput = activeCollection === "thinkers" ? wikipediaInput : detailsInput;
  if (detailInput) detailInput.value = entry?.[config.detailField] ?? "";
  showError(formError);
}

function thinkerTimeline(name) {
  const match = name.match(
    /^(.*?)\s+\((\d{1,4}(?: BC)?)(?:-(\d{1,4}(?: BC)?))?\)(?:\s+\(active (\d{4})-(\d{4})\))?$/i,
  );
  if (!match) return null;

  const birthUsesBC = [match[2], match[3]].some(
    (year) => year && /\s+BC$/i.test(year),
  );
  const parseYear = (year) =>
    Number(year.replace(/\s+BC$/i, "")) *
    (birthUsesBC || /\s+BC$/i.test(year) ? -1 : 1);
  const birthStart = parseYear(match[2]);
  const birthEnd = match[3] ? parseYear(match[3]) : null;
  const activeStart = match[4] ? Number(match[4]) : null;
  const activeEnd = match[5] ? Number(match[5]) : null;
  const meanYear = activeStart && activeEnd
    ? (activeStart + activeEnd) / 2
    : birthEnd
      ? (birthStart + birthEnd) / 2
      : birthStart;

  return {
    name: match[1],
    birthYears: birthEnd ? `${match[2]}-${match[3]}` : match[2],
    activeYears: activeStart && activeEnd ? `${match[4]}-${match[5]}` : null,
    meanYear,
  };
}

function makeEntryCard(entry, position) {
  const article = document.createElement("article");
  article.className = "entry-card";
  article.dataset.searchText = entry.name.toLocaleLowerCase();

  const index = document.createElement("span");
  index.className = "entry-index";
  const timeline = activeCollection === "thinkers"
    ? thinkerTimeline(entry.name)
    : null;
  const topicId = activeCollection === "concepts"
    ? entry.topic_assignments?.find((assignment) =>
      Number.isInteger(assignment.topic_id),
    )?.topic_id
    : null;
  index.textContent = timeline
    ? `~${Math.round(timeline.meanYear)}`
    : activeCollection === "thinkers"
      ? "—"
      : Number.isInteger(topicId)
        ? topicId < 0
          ? "—"
          : String(topicId)
        : String(position);

  const content = document.createElement("div");
  content.className = "entry-content";
  const title = document.createElement("h3");
  title.textContent = timeline ? timeline.name : entry.name;
  content.append(title);
  if (timeline) {
    const approximateYear = Math.round(timeline.meanYear);
    index.textContent = `~${Math.abs(approximateYear)}${approximateYear < 0 ? " BC" : ""}`;
    const dates = document.createElement("p");
    dates.className = "entry-dates";
    dates.textContent = [
      timeline.birthYears,
      timeline.activeYears ? `(active ${timeline.activeYears})` : null,
    ].filter(Boolean).join(" ");
    content.append(dates);
  }

  const detail = entry[collections[activeCollection].detailField];
  if (detail || activeCollection === "thinkers") {
    const description = document.createElement("p");
    if (activeCollection === "thinkers") {
      const links = [];
      if (entry.wiki_text_document_id) {
        const articleText = document.createElement("a");
        articleText.href =
          `/documents/${encodeURIComponent(entry.wiki_text_document_id)}`;
        articleText.target = "_blank";
        articleText.rel = "noopener noreferrer";
        articleText.textContent = "Wikipedia";
        links.push(articleText);
      }
      if (entry.dbpedia_document_id) {
        const dbpediaPage = document.createElement("a");
        dbpediaPage.href = `/documents/${encodeURIComponent(entry.dbpedia_document_id)}`;
        dbpediaPage.target = "_blank";
        dbpediaPage.rel = "noopener noreferrer";
        dbpediaPage.textContent = "DBpedia";
        links.push(dbpediaPage);
      }
      if (entry.military_thinking_document_id) {
        const militaryThinking = document.createElement("a");
        militaryThinking.href =
          `/documents/${encodeURIComponent(entry.military_thinking_document_id)}`;
        militaryThinking.target = "_blank";
        militaryThinking.rel = "noopener noreferrer";
        militaryThinking.textContent = "Ollama";
        links.push(militaryThinking);
      }
      for (const [index, link] of links.entries()) {
        if (index > 0) description.append(document.createTextNode(" · "));
        description.append(link);
      }
      if (links.length > 0) content.append(description);
    } else if (detail) {
      description.textContent = detail;
      content.append(description);
    }
  }
  if (activeCollection === "concepts") {
    // Hide the thinkers list for unclassified remainder topics
    const isUnclassified = entry.name && entry.name.trim() === "Unclassified topic (remainder)";
    if (!isUnclassified) {
      const assignments = document.createElement("div");
      assignments.className = "topic-assignments";
      const assignmentList = document.createElement("ul");
      assignmentList.className = "concept-thinkers";
      const assignmentThinkers = new Map(
        (entry.thinkers ?? []).map((thinker) => [thinker.id, thinker]),
      );
      const topicAssignments = entry.topic_assignments?.length
        ? entry.topic_assignments
        : (entry.thinkers ?? []).map((thinker) => ({
          thinker_id: thinker.id,
          topic_id: null,
        }));
      const orderedAssignments = [...topicAssignments].sort((left, right) => {
        const leftProbability = Number.isFinite(left.probability)
          ? left.probability
          : -1;
        const rightProbability = Number.isFinite(right.probability)
          ? right.probability
          : -1;
        return rightProbability - leftProbability;
      });
      for (const assignment of orderedAssignments) {
        const thinker = assignmentThinkers.get(assignment.thinker_id);
        if (!thinker) continue;
        const item = document.createElement("li");
        const sourceDocumentId =
          assignment.topic_source === "wikipedia"
            ? thinker.wiki_text_document_id
            : thinker.military_thinking_document_id;
        if (sourceDocumentId) {
          const link = document.createElement("a");
          link.href = `/documents/${encodeURIComponent(sourceDocumentId)}`;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = thinkerTimeline(thinker.name)?.name ?? thinker.name;
          item.append(link);
        } else {
          item.textContent = thinkerTimeline(thinker.name)?.name ?? thinker.name;
        }
        if (Number.isFinite(assignment.probability)) {
          item.append(
            document.createTextNode(
              ` (${Math.round(assignment.probability * 100)}%)`,
            ),
          );
        }
        assignmentList.append(item);
      }
      if (assignmentList.childElementCount > 0) {
        assignments.append(assignmentList);
      } else {
        const emptyAssignments = document.createElement("p");
        emptyAssignments.className = "topic-assignments-empty";
        emptyAssignments.textContent = "No topic assignments.";
        assignments.append(emptyAssignments);
      }
      content.append(assignments);
    }
  }

  const actions = document.createElement("div");
  actions.className = "entry-actions";

  if (activeCollection === "thinkers") {
    const reprocessButton = document.createElement("button");
    reprocessButton.className = "icon-button reprocess-button";
    reprocessButton.type = "button";
    reprocessButton.textContent = "Reprocess";
    reprocessButton.setAttribute("aria-label", `Reprocess ${entry.name}`);
    reprocessButton.addEventListener("click", () =>
      reprocessThinker(entry, reprocessButton, article),
    );
    actions.append(reprocessButton);

    const deleteButton = document.createElement("button");
    deleteButton.className = "icon-button delete-button";
    deleteButton.type = "button";
    deleteButton.textContent = "Delete";
    deleteButton.setAttribute("aria-label", `Delete ${entry.name}`);
    deleteButton.addEventListener("click", () => removeEntry(entry));
    actions.append(deleteButton);
  }
  article.append(index, content);
  if (activeCollection === "thinkers") article.append(actions);
  return article;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.error || `Request failed (${response.status}).`);
  }
  return response.status === 204 ? null : response.json();
}

async function loadCollectionCounts() {
  try {
    const [thinkers, concepts] = await Promise.all([
      requestJson("/api/thinkers"),
      requestJson("/api/concepts"),
    ]);
    document.querySelector("#thinker-count").textContent = String(thinkers.length);
    document.querySelector("#concept-count").textContent = String(concepts.length);
    setConnectionStatus(apiStatus, true, "API connected");
  } catch (error) {
    setConnectionStatus(apiStatus, false, "API unavailable");
    showError(listError, error.message);
  }
}

async function loadEntries() {
  const config = collections[activeCollection];
  const otherCollection = activeCollection === "concepts" ? "thinkers" : "concepts";
  showError(listError);
  list.replaceChildren();
  try {
    const [entries, otherEntries] = await Promise.all([
      requestJson(`/api/${activeCollection}`),
      requestJson(`/api/${otherCollection}`),
    ]);
    conceptCount = activeCollection === "concepts"
      ? entries.length
      : otherEntries.length;
    topicSearchBox.hidden =
      activeCollection !== "concepts" || Boolean(editingId) || conceptCount < 1;
    document.querySelector("#entry-total").textContent =
      `${entries.length} ${entries.length === 1 ? (activeCollection === "concepts" ? "topic" : "entry") : (activeCollection === "concepts" ? "topics" : "entries")}`;
    document.querySelector(`#${activeCollection === "concepts" ? "concept" : "thinker"}-count`).textContent =
      String(entries.length);
    document.querySelector(`#${otherCollection === "concepts" ? "concept" : "thinker"}-count`).textContent =
      String(otherEntries.length);
    setConnectionStatus(apiStatus, true, "API connected");
    const orderedEntries = activeCollection === "thinkers"
      ? entries
        .map((entry, originalIndex) => ({
          entry,
          meanYear: thinkerTimeline(entry.name)?.meanYear ?? null,
          originalIndex,
        }))
        .sort((left, right) => {
          if (left.meanYear === null) {
            return right.meanYear === null
              ? left.originalIndex - right.originalIndex
              : -1;
          }
          return right.meanYear === null
            ? 1
            : left.meanYear - right.meanYear;
        })
        .map(({ entry }) => entry)
      : entries
        .map((entry, originalIndex) => ({
          entry,
          topicId: entry.topic_assignments?.find((assignment) =>
            Number.isInteger(assignment.topic_id) && assignment.topic_id >= 0,
          )?.topic_id ?? null,
          originalIndex,
        }))
        .sort((left, right) => {
          if (left.topicId === null) {
            return right.topicId === null
              ? left.originalIndex - right.originalIndex
              : 1;
          }
          return right.topicId === null
            ? -1
            : left.topicId - right.topicId || left.originalIndex - right.originalIndex;
        })
        .map(({ entry }) => entry);
    if (orderedEntries.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "Nothing here yet. Generate topics first.";
      list.append(empty);
    } else {
      for (const [index, entry] of orderedEntries.entries()) {
        list.append(
          makeEntryCard(
            entry,
            activeCollection === "concepts" ? index : index + 1,
          ),
        );
      }
    }
    applyThinkerSearch();
  } catch (error) {
    setConnectionStatus(apiStatus, false, "API unavailable");
    showError(listError, error.message);
  }
}

function applyThinkerSearch() {
  if (activeCollection !== "thinkers") return;
  const query = thinkerSearch.value;
  const cards = [...list.querySelectorAll(".entry-card")];
  let visibleCount = 0;
  for (const card of cards) {
    const visible = searchMatches(card.dataset.searchText, query);
    card.hidden = !visible;
    if (visible) visibleCount += 1;
  }
  let emptyState = list.querySelector(".search-empty-state");
  if (cards.length > 0 && visibleCount === 0) {
    if (!emptyState) {
      emptyState = document.createElement("p");
      emptyState.className = "empty-state search-empty-state";
      list.append(emptyState);
    }
    emptyState.textContent = "No thinkers match your search.";
  } else {
    emptyState?.remove();
  }
  document.querySelector("#entry-total").textContent = query
    ? `${visibleCount} of ${cards.length} thinkers`
    : `${cards.length} ${cards.length === 1 ? "thinker" : "thinkers"}`;
}

async function checkLanguageModel() {
  try {
    const status = await requestJson("/api/status/language-model");
    setConnectionStatus(
      languageModelStatus,
      status.connected,
      status.connected ? "LLM connected" : "LLM unavailable",
    );
  } catch {
    setConnectionStatus(languageModelStatus, false, "LLM unavailable");
  }
}

async function removeEntry(entry) {
  const config = collections[activeCollection];
  if (!window.confirm(`Delete "${entry.name}"? This cannot be undone.`)) return;
  try {
    await requestJson(`/api/${activeCollection}/${entry.id}`, { method: "DELETE" });
    if (editingId === entry.id) setEditing();
    await loadEntries();
  } catch (error) {
    showError(listError, error.message);
  }
}

async function reprocessThinker(entry, button, article) {
  if (!entry.biography) {
    showError(listError, `Cannot reprocess "${entry.name}": Wikipedia URL is missing.`);
    return;
  }

  const originalLabel = button.textContent;
  const tabs = [...document.querySelectorAll(".tab")];
  button.disabled = true;
  button.textContent = "Reprocessing…";
  article.setAttribute("aria-busy", "true");
  tabs.forEach((tab) => {
    tab.disabled = true;
  });
  showError(listError);
  try {
    await requestJson(`/api/thinkers/${entry.id}/process`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        biography: entry.biography,
        prompt: ollamaPromptInput.value,
        word_count: Number(responseWordCountInput.value),
      }),
    });
    await loadEntries();
  } catch (error) {
    showError(listError, error.message);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
    article.removeAttribute("aria-busy");
    tabs.forEach((tab) => {
      tab.disabled = false;
    });
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (saveButton.disabled) return;
  const config = collections[activeCollection];
  const isGeneratingConcepts = activeCollection === "concepts" && !editingId;
  const detailInput = activeCollection === "thinkers" ? wikipediaInput : detailsInput;
  const payload = isGeneratingConcepts ? {} : {
    [config.detailField]: detailInput.value,
  };
  if (isGeneratingConcepts && topicCountInput.value !== "") {
    payload.topic_count = Number(topicCountInput.value);
  }
  if (isGeneratingConcepts) {
    payload.topic_source = topicSourceSelect.value;
    payload.min_df = Number(topicMinDfInput.value);
    payload.max_words_in_topic = Number(topicMaxWordsInput.value);
    payload.top_n_words = Number(topicTopNWordsInput.value);
    payload.diversity = Number(topicDiversityInput.value);
  }
  if (activeCollection === "thinkers" && !editingId) {
  payload.word_count = Number(responseWordCountInput.value);
  payload.prompt = ollamaPromptInput.value;
  }
  if (activeCollection === "concepts" && editingId) payload.name = nameInput.value;
  const url = activeCollection === "thinkers"
    ? `/api/thinkers${editingId ? `/${editingId}` : ""}/process`
    : isGeneratingConcepts
      ? "/api/concepts/generate"
      : `/api/concepts/${editingId}`;
  const originalButtonContent = saveButton.innerHTML;
  const isProcessingThinker = activeCollection === "thinkers";
  saveButton.disabled = true;
  saveButton.classList.add("is-processing");
  saveButton.innerHTML =
    '<span class="processing-spinner" aria-hidden="true"></span>' +
    `${isProcessingThinker ? "Processing thinker…" : "Generating topics…"}`;
  wikipediaInput.disabled = true;
  responseWordCountInput.disabled = true;
  ollamaPromptInput.disabled = true;
  topicSourceSelect.disabled = true;
  nameInput.disabled = true;
  detailsInput.disabled = true;
  topicCountInput.disabled = true;
  topicMinDfInput.disabled = true;
  topicMaxWordsInput.disabled = true;
  topicTopNWordsInput.disabled = true;
  topicDiversityInput.disabled = true;
  topicSearchButton.disabled = true;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.disabled = true;
  });
  form.setAttribute("aria-busy", "true");
  formStatus.replaceChildren();
  const statusSpinner = document.createElement("span");
  statusSpinner.className = "processing-spinner";
  statusSpinner.setAttribute("aria-hidden", "true");
  formStatus.append(
    statusSpinner,
    document.createTextNode(
      isProcessingThinker
        ? "Downloading sources and analyzing with the language model. This can take a few minutes; keep this page open."
        : isGeneratingConcepts
          ? `Analyzing all processed ${topicSourceSelect.value === "ollama" ? "Ollama analyses" : "Wikipedia texts"} with BERTopic. This can take a few minutes; keep this page open.`
          : "Saving topic changes.",
    ),
  );
  formStatus.hidden = false;
  showError(formError);
  let succeeded = false;
  try {
    await requestJson(url, {
      method: editingId ? "PATCH" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (isGeneratingConcepts) {
      topicSearchResults.replaceChildren();
      showError(topicSearchError);
    }
    setEditing();
    await loadEntries();
    succeeded = true;
  } catch (error) {
    showError(formError, error.message);
    if (error.message.toLowerCase().includes("language model is unavailable")) {
      setConnectionStatus(
        languageModelStatus,
        false,
        "LLM unavailable",
      );
    }
  } finally {
    saveButton.disabled = false;
    saveButton.classList.remove("is-processing");
    wikipediaInput.disabled = false;
    responseWordCountInput.disabled = false;
    ollamaPromptInput.disabled = false;
    nameInput.disabled = false;
    detailsInput.disabled = false;
    topicCountInput.disabled = false;
    const topicOptionsDisabled =
      activeCollection !== "concepts" || Boolean(editingId);
    topicMinDfInput.disabled = topicOptionsDisabled;
    topicMaxWordsInput.disabled = topicOptionsDisabled;
    topicTopNWordsInput.disabled = topicOptionsDisabled;
    topicDiversityInput.disabled = topicOptionsDisabled;
    topicSearchButton.disabled = false;
    topicSourceSelect.disabled = false;
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.disabled = false;
    });
    form.removeAttribute("aria-busy");
    formStatus.hidden = true;
    if (!succeeded) saveButton.innerHTML = originalButtonContent;
  }
});

topicSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (topicSearchButton.disabled) return;
  topicSearchResults.replaceChildren();
  showError(topicSearchError);
  topicSearchButton.disabled = true;
  topicSearchButton.textContent = "Searching…";
  topicSearchForm.setAttribute("aria-busy", "true");
  try {
    const response = await requestJson("/api/concepts/find-topics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        concept: topicSearchConceptInput.value,
      }),
    });
    const relatedTopics = response.results.filter(
      (result) => Math.round(result.similarity * 100) > 0,
    );
    if (relatedTopics.length === 0) {
      const emptyResult = document.createElement("li");
      emptyResult.textContent = "No related topics were found.";
      topicSearchResults.append(emptyResult);
    }
    for (const result of relatedTopics) {
      const item = document.createElement("li");
      const topicNumber = document.createElement("span");
      topicNumber.className = "entry-index";
      topicNumber.textContent = result.topic_id < 0
        ? "—"
        : String(result.topic_id);
      const name = document.createElement("strong");
      name.textContent = result.name;
      item.append(topicNumber, name);
      const similarityPercent = Math.round(result.similarity * 100);
      if (similarityPercent > 0) {
        const similarity = document.createElement("span");
        similarity.className = "topic-search-similarity";
        similarity.textContent = `${similarityPercent}%`;
        item.append(similarity);
      }
      topicSearchResults.append(item);
    }
  } catch (error) {
    showError(topicSearchError, error.message);
  } finally {
    topicSearchButton.disabled = false;
    topicSearchButton.innerHTML = 'Find topics <span aria-hidden="true">→</span>';
    topicSearchForm.removeAttribute("aria-busy");
  }
});

cancelButton.addEventListener("click", () => setEditing());
thinkerSearch.addEventListener("input", applyThinkerSearch);
distributionThinkerSearch.addEventListener("input", () => {
  updateTopicDistributionBubbles();
  topicsVisualization.contentWindow?.postMessage(
    {
      type: "thinker-search",
      query: distributionThinkerSearch.value,
    },
    "*",
  );
});
visualizationOptions.forEach((option) => {
  option.addEventListener("click", () => {
    visualizationOptions.forEach((item) => {
      const selected = item === option;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
    distributionRetry.addEventListener("click", loadTopicDistributions);
    loadVisualization();
    if (option.dataset.visualization === "approximate-distribution") {
      loadTopicDistributions();
    }
  });
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", async () => {
    const collection = tab.dataset.collection;
    const showVisualizations = collection === "visualizations";
    const showAbout = collection === "about";

    // Show/hide main workspace, visualizations panel, and about panel appropriately
    workspace.hidden = showVisualizations || showAbout;
    visualizationsPanel.hidden = !showVisualizations;
    const aboutPanel = document.querySelector("#about-panel");
    if (aboutPanel) aboutPanel.hidden = !showAbout;

    document.querySelectorAll(".tab").forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });

    if (showVisualizations) {
      loadVisualization();
      loadTopicDistributions();
      return;
    }

    if (showAbout) {
      loadCollectionCounts();
      return;
    }

    // For Thinkers/Topics, set the active collection and refresh entries
    activeCollection = collection;
    document.querySelector("#entries-heading").textContent = collections[activeCollection].plural;
    nameInput.placeholder = activeCollection === "concepts"
      ? "e.g. maneuver warfare"
      : "Carl van Clausewitz";
    setEditing();
    loadEntries();
  });
});

// Default to showing the About tab on first load. If About tab exists, trigger its click
// handler to show the About panel without attempting to load entries for thinkers/topics.
const aboutTabButton = document.querySelector('.tab[data-collection="about"]');
if (aboutTabButton) {
  // Use a microtask to ensure all tab handlers are attached before invoking click
  Promise.resolve().then(() => aboutTabButton.click());
} else {
  // Fallback to original behavior when About tab isn't present
  loadEntries();
  setEditing();
}

checkLanguageModel();
window.setInterval(checkLanguageModel, 30000);
