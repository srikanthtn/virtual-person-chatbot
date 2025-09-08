const apiUrl = "http://127.0.0.1:8000"; // Adjust the URL if needed

document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const queryForm = document.getElementById("query-form");

    // Handle upload form
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch(`${apiUrl}/memories/upload`, {
                method: "POST",
                body: formData,
            });
            
            if (response.ok) {
                const result = await response.json();
                alert(`Memory uploaded successfully!\nMemory ID: ${result.memory_id}\nUUID: ${result.uuid}`);
                uploadForm.reset();
            } else {
                const error = await response.json();
                alert(`Upload failed: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            alert(`Upload failed: ${error.message}`);
        }
    });

    // Handle query form
    queryForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const query = document.getElementById('query').value.trim();
        const top_k = parseInt(document.getElementById('top_k').value, 10) || 5;

        if (!query) {
            alert('Please enter a search query.');
            return;
        }

        // Show loading state
        showLoading(true);
        hideResults();

        try {
            const response = await fetch(`${apiUrl}/memories/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k })
            });

            if (response.ok) {
                const data = await response.json();
                displayQueryResults(data);
            } else {
                const error = await response.json();
                showError(`Query failed: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            showError(`Query failed: ${error.message}`);
        } finally {
            showLoading(false);
        }
    });

    function showLoading(show) {
        const loading = document.getElementById('loading');
        loading.style.display = show ? 'block' : 'none';
    }

    function hideResults() {
        document.getElementById('answer-section').style.display = 'none';
        document.getElementById('results-section').style.display = 'none';
    }

    function showError(message) {
        const answerSection = document.getElementById('answer-section');
        const llmAnswer = document.getElementById('llm-answer');
        
        answerSection.style.display = 'block';
        llmAnswer.innerHTML = `<div style="color: #d32f2f; background: #ffebee; border: 1px solid #f8bbd9; padding: 15px; border-radius: 6px;">${message}</div>`;
    }

    function displayQueryResults(data) {
        // Display LLM answer
        const answerSection = document.getElementById('answer-section');
        const llmAnswer = document.getElementById('llm-answer');
        
        answerSection.style.display = 'block';
        llmAnswer.innerHTML = data.llm_answer || "No answer generated.";

        // Display ChromaDB results
        const resultsSection = document.getElementById('results-section');
        const queryMemories = document.getElementById('query-memories');
        
        if (data.results && data.results.length > 0) {
            resultsSection.style.display = 'block';
            queryMemories.innerHTML = data.results.map(memory => createMemoryCard(memory)).join('');
        } else {
            resultsSection.style.display = 'block';
            queryMemories.innerHTML = '<div style="text-align: center; color: #666; padding: 20px; font-style: italic;">No relevant memories found.</div>';
        }
    }

    function createMemoryCard(memory) {
        const content = memory.transcript || memory.caption || memory.text || 'No content available';
        const date = new Date(memory.created_at).toLocaleDateString();
        const time = new Date(memory.created_at).toLocaleTimeString();
        
        return `
            <div class="memory-item">
                <div class="memory-header">
                    <span class="memory-type">${memory.kind.toUpperCase()}</span>
                    <span class="memory-date">${date} at ${time}</span>
                </div>
                <div class="memory-content">
                    ${content}
                </div>
                ${memory.filename ? `<div style="margin-top: 8px; font-size: 12px; color: #666;">File: ${memory.filename}</div>` : ''}
            </div>
        `;
    }
});