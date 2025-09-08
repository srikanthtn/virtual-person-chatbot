const apiUrl = "http://127.0.0.1:8000"; // Adjust the URL if needed

document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const queryForm = document.getElementById("query-form");
    const resultsContainer = document.getElementById("results");

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        const response = await fetch(`${apiUrl}/memories/upload`, {
            method: "POST",
            body: formData,
        });
        const result = await response.json();
        alert(`Memory uploaded with ID: ${result.memory_id}`);
        uploadForm.reset();
    });

    queryForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const queryText = document.getElementById("query-text").value;
        const topK = document.getElementById("top-k").value;

        const response = await fetch(`${apiUrl}/memories/query`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query: queryText, top_k: topK }),
        });
        const results = await response.json();
        displayResults(results.results);
    });

    function displayResults(results) {
        resultsContainer.innerHTML = "";
        results.forEach((memory) => {
            const memoryDiv = document.createElement("div");
            memoryDiv.innerHTML = `
                <h3>${memory.kind} Memory</h3>
                <p>ID: ${memory.memory_id}</p>
                <p>UUID: ${memory.uuid}</p>
                <p>Created At: ${memory.created_at}</p>
                <p>Transcript: ${memory.transcript || "N/A"}</p>
                <p>Caption: ${memory.caption || "N/A"}</p>
                <p>Filename: ${memory.filename || "N/A"}</p>
                <hr>
            `;
            resultsContainer.appendChild(memoryDiv);
        });
    }
});

document.getElementById('query-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const query = document.getElementById('query').value;
    const top_k = parseInt(document.getElementById('top_k').value, 10) || 5;

    document.getElementById('llm-answer').textContent = "Loading...";
    document.getElementById('query-memories').textContent = "";

    const response = await fetch('/memories/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k })
    });

    if (response.ok) {
        const data = await response.json();
        document.getElementById('llm-answer').innerHTML = `<strong>Answer:</strong> ${data.llm_answer || "No answer."}`;
        if (data.results && data.results.length > 0) {
            let memList = "<h3>Relevant Memories:</h3><ul>";
            data.results.forEach(mem => {
                memList += `<li>${mem.kind} - ${mem.transcript || mem.caption || mem.text || ''}</li>`;
            });
            memList += "</ul>";
            document.getElementById('query-memories').innerHTML = memList;
        } else {
            document.getElementById('query-memories').innerHTML = "<em>No relevant memories found.</em>";
        }
    } else {
        document.getElementById('llm-answer').textContent = "Error fetching answer.";
    }
});