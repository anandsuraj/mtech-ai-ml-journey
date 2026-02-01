// RAG System Frontend JavaScript
// Handles user interactions and API calls

let isLoading = false;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    const searchBtn = document.getElementById('searchBtn');
    const searchInput = document.getElementById('searchInput');
    
    // Handle search button click
    searchBtn.addEventListener('click', performSearch);
    
    // Handle enter key in search box
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !isLoading) {
            performSearch();
        }
    });
});

// Main search function
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        alert('Please enter a question');
        return;
    }
    
    if (isLoading) return;
    
    // Show loading state
    setLoadingState(true);
    hideResults();
    
    try {
        // Call the search API
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

// Show/hide loading state
function setLoadingState(loading) {
    isLoading = loading;
    const searchBtn = document.getElementById('searchBtn');
    const loadingDiv = document.getElementById('loading');
    
    if (loading) {
        searchBtn.disabled = true;
        searchBtn.textContent = 'Searching...';
        loadingDiv.classList.add('active');
    } else {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Search';
        loadingDiv.classList.remove('active');
    }
}

// Hide previous results
function hideResults() {
    document.getElementById('answerSection').classList.remove('active');
    document.getElementById('chunksSection').classList.remove('active');
}

// Display search results
function displayResults(data) {
    // Show answer
    document.getElementById('answerText').textContent = data.answer;
    document.getElementById('answerSection').classList.add('active');
    
    // Show metrics
    document.getElementById('retrievalTime').textContent = data.retrieval_time.toFixed(2) + 's';
    document.getElementById('generationTime').textContent = data.generation_time.toFixed(2) + 's';
    document.getElementById('totalTime').textContent = data.total_time.toFixed(2) + 's';
    document.getElementById('chunkCount').textContent = data.chunks.length;
    
    // Display retrieved chunks
    displayChunks(data.chunks);
    
    // Scroll to results
    document.getElementById('answerSection').scrollIntoView({ behavior: 'smooth' });
}

// Display retrieved chunks
function displayChunks(chunks) {
    const container = document.getElementById('chunksContainer');
    container.innerHTML = '';
    
    chunks.forEach((chunk, index) => {
        const chunkCard = createChunkCard(chunk, index + 1);
        container.appendChild(chunkCard);
    });
    
    document.getElementById('chunksSection').classList.add('active');
}

// Create chunk card HTML
function createChunkCard(chunk, rank) {
    const card = document.createElement('div');
    card.className = 'chunk-card';
    
    card.innerHTML = `
        <div class="chunk-header">
            <div class="chunk-title">${escapeHtml(chunk.title)}</div>
            <div class="chunk-rank">#${rank}</div>
        </div>
        <a href="${escapeHtml(chunk.url)}" target="_blank" class="chunk-url">${escapeHtml(chunk.url)}</a>
        <div class="chunk-scores">
            <span class="score-badge score-rrf">RRF: ${chunk.rrf_score.toFixed(4)}</span>
            ${chunk.dense_rank ? `<span class="score-badge score-dense">Dense Rank: ${chunk.dense_rank}</span>` : ''}
            ${chunk.sparse_rank ? `<span class="score-badge score-sparse">Sparse Rank: ${chunk.sparse_rank}</span>` : ''}
        </div>
        <div class="chunk-text">${escapeHtml(chunk.text)}</div>
    `;
    
    return card;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
