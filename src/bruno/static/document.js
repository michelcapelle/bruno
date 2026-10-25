const editor = document.getElementById("document-text-editor");
const saveButton = document.getElementById("document-save-button");
const saveStatus = document.getElementById("document-save-status");

if (editor && saveButton && saveStatus) {
  saveButton.addEventListener("click", async () => {
    saveButton.disabled = true;
    saveStatus.textContent = "Saving…";
    try {
      const response = await fetch(
        `/api/documents/${encodeURIComponent(editor.dataset.documentId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: editor.value }),
        },
      );
      const result = await response.json();
      saveStatus.textContent = response.ok
        ? "Saved."
        : result.error || "Could not save the text.";
    } catch (error) {
      saveStatus.textContent = "Could not save the text.";
    } finally {
      saveButton.disabled = false;
    }
  });
}
