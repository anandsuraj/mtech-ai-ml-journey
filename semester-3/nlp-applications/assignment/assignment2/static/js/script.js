/**
 * Statistical Machine Translation - Frontend Logic
 * NLP Applications Assignment 2
 * 
 * Handles user interactions, API calls, and result display
 */

// Global state
let translatedText = '';
let currentTab = 'manual';

// DOM Elements
const sourceText = document.getElementById('source-text');
const sourceLang = document.getElementById('source-lang');
const targetLang = document.getElementById('target-lang');
const translateBtn = document.getElementById('translate-btn');
const evaluateBtn = document.getElementById('evaluate-btn');
const addReferenceBtn = document.getElementById('add-reference-btn');
const resultsSection = document.getElementById('results-section');
const bleuSection = document.getElementById('bleu-section');
const errorMessage = document.getElementById('error-message');

// Tab switching
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active states
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        currentTab = tabName;
    });
});

// Add reference input field
addReferenceBtn.addEventListener('click', () => {
    const referenceInputs = document.getElementById('reference-inputs');
    const currentCount = referenceInputs.children.length;
    
    const newInputGroup = document.createElement('div');
    newInputGroup.className = 'reference-input-group';
    newInputGroup.innerHTML = `
        <label>Reference ${currentCount + 1}:</label>
        <textarea class="reference-input" rows="3" placeholder="Enter reference translation..."></textarea>
    `;
    
    referenceInputs.appendChild(newInputGroup);
});

// File upload handling
const referenceFile = document.getElementById('reference-file');
referenceFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const text = await file.text();
        const lines = text.split('\n').filter(line => line.trim());
        
        // Clear existing inputs and add new ones
        const referenceInputs = document.getElementById('reference-inputs');
        referenceInputs.innerHTML = '';
        
        lines.forEach((line, index) => {
            const inputGroup = document.createElement('div');
            inputGroup.className = 'reference-input-group';
            inputGroup.innerHTML = `
                <label>Reference ${index + 1}:</label>
                <textarea class="reference-input" rows="3">${line.trim()}</textarea>
            `;
            referenceInputs.appendChild(inputGroup);
        });
        
        // Switch to manual tab to show loaded references
        tabButtons[0].click();
        
        showSuccess(`Loaded ${lines.length} reference(s) from file`);
    }
});

// Translation function
translateBtn.addEventListener('click', async () => {
    const source = sourceText.value.trim();
    const srcLang = sourceLang.value;
    const tgtLang = targetLang.value;
    
    // Validation
    if (!source) {
        showError('Please enter text to translate');
        return;
    }
    
    if (srcLang === tgtLang) {
        showError('Source and target languages must be different');
        return;
    }
    
    // Show loading state
    setButtonLoading(translateBtn, true);
    hideError();
    
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                source_text: source,
                source_lang: srcLang,
                target_lang: tgtLang
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            translatedText = data.translated_text;
            displayTranslation(translatedText);
            evaluateBtn.disabled = false;
        } else {
            showError(data.error || 'Translation failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        setButtonLoading(translateBtn, false);
    }
});

// BLEU evaluation function
evaluateBtn.addEventListener('click', async () => {
    if (!translatedText) {
        showError('Please translate text first');
        return;
    }
    
    // Get reference translations
    const references = getReferenceTranslations();
    
    if (references.length === 0) {
        showError('Please provide at least one reference translation');
        return;
    }
    
    // Show loading state
    setButtonLoading(evaluateBtn, true);
    hideError();
    
    try {
        const response = await fetch('/evaluate_bleu', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                candidate: translatedText,
                references: references
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayBLEUScore(data);
        } else {
            showError(data.error || 'BLEU evaluation failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        setButtonLoading(evaluateBtn, false);
    }
});

// Helper function to get reference translations from inputs
function getReferenceTranslations() {
    const referenceInputs = document.querySelectorAll('.reference-input');
    const references = [];
    
    referenceInputs.forEach(input => {
        const text = input.value.trim();
        if (text) {
            references.push(text);
        }
    });
    
    return references;
}

// Display translation result
function displayTranslation(translation) {
    document.getElementById('translated-text').textContent = translation;
    resultsSection.style.display = 'block';
    bleuSection.style.display = 'none';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display BLEU score and metrics
function displayBLEUScore(data) {
    // Show BLEU section
    bleuSection.style.display = 'block';
    
    // Display BLEU score
    const bleuScore = data.bleu_score;
    document.getElementById('bleu-score').textContent = bleuScore.toFixed(4);
    
    // Quality assessment based on BLEU score
    const qualityElement = document.getElementById('bleu-quality');
    let quality, color;
    
    if (bleuScore < 0.3) {
        quality = 'Poor Quality';
        color = '#ef4444';
    } else if (bleuScore < 0.5) {
        quality = 'Fair Quality';
        color = '#f59e0b';
    } else if (bleuScore <  0.7) {
        quality = 'Good Quality';
        color = '#eab308';
    } else {
        quality = 'Excellent Quality';
        color = '#10b981';
    }
    
    qualityElement.textContent = quality;
    qualityElement.style.color = color;
    
    // Display n-gram precision table
    const tbody = document.getElementById('ngram-tbody');
    tbody.innerHTML = '';
    
    const precisionDetails = data.precision_details;
    Object.entries(precisionDetails).forEach(([ngram, precision]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${ngram}</td>
            <td>${precision.toFixed(4)}</td>
            <td>${(precision * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(row);
    });
    
    // Display additional metrics
    document.getElementById('brevity-penalty').textContent = data.brevity_penalty.toFixed(4);
    document.getElementById('candidate-length').textContent = data.candidate_length;
    document.getElementById('reference-length').textContent = data.reference_length;
    
    // Scroll to BLEU section
    bleuSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Show success message
function showSuccess(message) {
    errorMessage.textContent = message;
    errorMessage.style.background = '#10b981';
    errorMessage.style.display = 'block';
    
    setTimeout(() => {
        errorMessage.style.display = 'none';
        errorMessage.style.background = '#ef4444';
    }, 3000);
}

// Set button loading state
function setButtonLoading(button, isLoading) {
    const btnText = button.querySelector('.btn-text');
    const spinner = button.querySelector('.spinner');
    
    if (isLoading) {
        btnText.style.display = 'none';
        spinner.style.display = 'inline-block';
        button.disabled = true;
    } else {
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
        button.disabled = false;
    }
}


