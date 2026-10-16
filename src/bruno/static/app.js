const collections = {
  concepts: {
    label: "Concept",
    plural: "Concepts",
    detailField: "description",
    detailLabel: "Description",
    placeholder: "What does this idea mean?",
  },
  thinkers: {
    label: "Thinker",
    plural: "Thinkers",
    detailField: "biography",
    detailLabel: "Biography",
    placeholder: "A little about this thinker...",
  },
};

const form = document.querySelector("#entry-form");
const nameInput = document.querySelector("#entry-name");
const detailsInput = document.querySelector("#entry-details");
const heading = document.querySelector("#form-heading");
const formEyebrow = document.querySelector("#form-eyebrow");
const detailsLabel = document.querySelector("#details-label");
const saveButton = document.querySelector("#save-button");
const cancelButton = document.querySelector("#cancel-button");
const list = document.querySelector("#entry-list");
const formError = document.querySelector("#form-error");
const listError = document.querySelector("#list-error");
const apiStatus = document.querySelector("#api-status");

let activeCollection = "concepts";
let editingId = null;

function showError(element, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function setEditing(entry = null) {
  editingId = entry ? entry.id : null;
  const config = collections[activeCollection];
  formEyebrow.textContent = entry ? "EDIT ENTRY" : "NEW ENTRY";
  heading.textContent = entry ? `Edit ${config.label.toLowerCase()}` : `Add a ${config.label.toLowerCase()}`;
  saveButton.innerHTML = entry
    ? 'Save changes <span aria-hidden="true">→</span>'
    : `Save ${config.label.toLowerCase()} <span aria-hidden="true">→</span>`;
  cancelButton.hidden = !entry;
  nameInput.value = entry?.name ?? "";
  detailsInput.value = entry?.[config.detailField] ?? "";
  showError(formError);
}

function makeEntryCard(entry) {
  const article = document.createElement("article");
  article.className = "entry-card";

  const index = document.createElement("span");
  index.className = "entry-index";
  index.textContent = String(entry.id).padStart(2, "0");

  const content = document.createElement("div");
  content.className = "entry-content";
  const title = document.createElement("h3");
  title.textContent = entry.name;
  content.append(title);

  const detail = entry[collections[activeCollection].detailField];
  if (detail) {
    const description = document.createElement("p");
    description.textContent = detail;
    content.append(description);
  }

  const actions = document.createElement("div");
  actions.className = "entry-actions";
  const editButton = document.createElement("button");
  editButton.className = "icon-button";
  editButton.type = "button";
  editButton.textContent = "Edit";
  editButton.setAttribute("aria-label", `Edit ${entry.name}`);
  editButton.addEventListener("click", () => setEditing(entry));

  const deleteButton = document.createElement("button");
  deleteButton.className = "icon-button delete-button";
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.setAttribute("aria-label", `Delete ${entry.name}`);
  deleteButton.addEventListener("click", () => removeEntry(entry));
  actions.append(editButton, deleteButton);
  article.append(index, content, actions);
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
    document.querySelector("#entry-total").textContent =
      `${entries.length} ${entries.length === 1 ? "entry" : "entries"}`;
    document.querySelector(`#${activeCollection === "concepts" ? "concept" : "thinker"}-count`).textContent =
      String(entries.length);
    document.querySelector(`#${otherCollection === "concepts" ? "concept" : "thinker"}-count`).textContent =
      String(otherEntries.length);
    apiStatus.lastChild.textContent = " API connected";
    apiStatus.classList.remove("unavailable");
    if (entries.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = `Nothing here yet. Add your first ${config.label.toLowerCase()} to get started.`;
      list.append(empty);
      return;
    }
    for (const entry of entries) list.append(makeEntryCard(entry));
  } catch (error) {
    apiStatus.lastChild.textContent = " API unavailable";
    apiStatus.classList.add("unavailable");
    showError(listError, error.message);
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const config = collections[activeCollection];
  const payload = {
    name: nameInput.value,
    [config.detailField]: detailsInput.value,
  };
  const url = `/api/${activeCollection}${editingId ? `/${editingId}` : ""}`;
  try {
    await requestJson(url, {
      method: editingId ? "PATCH" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setEditing();
    await loadEntries();
  } catch (error) {
    showError(formError, error.message);
  }
});

cancelButton.addEventListener("click", () => setEditing());

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    activeCollection = tab.dataset.collection;
    document.querySelectorAll(".tab").forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    document.querySelector("#entries-heading").textContent = collections[activeCollection].plural;
    detailsLabel.innerHTML = `${collections[activeCollection].detailLabel} <span>Optional</span>`;
    detailsInput.placeholder = collections[activeCollection].placeholder;
    nameInput.placeholder = activeCollection === "concepts"
      ? "e.g. The public sphere"
      : "e.g. Hannah Arendt";
    setEditing();
    loadEntries();
  });
});

loadEntries();
