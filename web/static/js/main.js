// Debounce function to limit API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// DOM elements
const contextInput = document.getElementById('context');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const predictionSpan = document.getElementById('prediction');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const usePredictBtn = document.getElementById('usePredict');
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeModal = document.getElementById('closeModal');
const alphaValue = document.getElementById('alphaValue');
const vocabSize = document.getElementById('vocabSize');
const perplexityStats = document.getElementById('perplexityStats');

// Validate input text
function validateInput(text) {
    // Allow letters (including Polish), and spaces
    const letterOnlyRegex = /^[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]*$/;
    return letterOnlyRegex.test(text);
}

// Clear input and results
function clearInput() {
    contextInput.value = '';
    hideResults();
    hideError();
    contextInput.focus();
}

// Hide prediction results
function hideResults() {
    resultDiv.classList.remove('show');
    predictionSpan.textContent = 'Type something to see prediction';
}

// Show loading state
function setLoading(isLoading) {
    if (isLoading) {
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
    } else {
        setTimeout(() => {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }, 300); // Small delay to ensure smooth transition
    }
}

// Show error message
function showError(message) {
    // Create or clear existing error div content
    errorDiv.innerHTML = `
        <div class="error-icon">
            <i class="fas fa-exclamation-circle"></i>
        </div>
        <div class="error-content">${message}</div>
        <button class="error-close">
            <i class="fas fa-times"></i>
        </button>
        <div class="error-progress"></div>
    `;
    
    // Add close button event listener
    const closeBtn = errorDiv.querySelector('.error-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            errorDiv.classList.remove('show');
        });
    }
    
    // Show the error
    errorDiv.classList.add('show');
    
    // Auto-dismiss after 5 seconds
    const autoDismissTimeout = setTimeout(() => {
        errorDiv.classList.remove('show');
    }, 5000);
    
    // Store the timeout ID so it can be cleared if the error is manually closed
    errorDiv._dismissTimeout = autoDismissTimeout;
}

// Hide error message
function hideError() {
    // Clear any existing timeout
    if (errorDiv._dismissTimeout) {
        clearTimeout(errorDiv._dismissTimeout);
        errorDiv._dismissTimeout = null;
    }
    
    errorDiv.classList.remove('show');
}

// Get prediction from API
async function getPrediction(isUserAction = false) {
    const context = contextInput.value.trim();
    
    // Validate input
    if (context && !validateInput(context)) {
        showError('Context can only contain letters and spaces.');
        return;
    }
    
    const words = context ? context.split(/\s+/) : [];
    
    // Clear previous results
    hideError();
    
    // For real-time predictions (not user action), require non-empty input
    if (!isUserAction && !context) {
        hideResults();
        return;
    }
    
    if (words.length > 3) {
        showError('Maximum context length is 3 words.');
        return;
    }
    
    try {
        setLoading(true);
        
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ context: words })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            hideError();
            predictionSpan.textContent = data.prediction;
            resultDiv.classList.add('show');
        }
    } catch (error) {
        showError('Could not connect to the server. Please try again.');
    } finally {
        setLoading(false);
    }
}

// Use the predicted word
function usePrediction() {
    const prediction = predictionSpan.textContent;
    if (prediction) {
        const currentText = contextInput.value;
        const newText = currentText ? `${currentText} ${prediction}` : prediction;
        contextInput.value = newText;
        hideResults();
        contextInput.focus();
    }
}

// Open settings modal and fetch model stats
async function openSettingsModal() {
    settingsModal.classList.add('show');
    await fetchModelStats();
}

// Close settings modal
function closeSettingsModal() {
    settingsModal.classList.remove('show');
}

// Fetch model statistics
async function fetchModelStats() {
    try {
        // Show loading state
        perplexityStats.innerHTML = `
            <div class="stat-loader">
                <div class="spinner"></div>
                <p>Loading statistics...</p>
            </div>
        `;
        
        const response = await fetch('http://localhost:8000/model-stats');
        const data = await response.json();
        
        // Update model info
        alphaValue.textContent = data.alpha;
        vocabSize.textContent = data.vocabulary_size;
        
        // Create perplexity cards
        let perplexityHtml = '';
        for (const [gramType, value] of Object.entries(data.perplexity)) {
            perplexityHtml += `
                <div class="perplexity-card">
                    <div class="ngram-label">${gramType}</div>
                    <div class="perplexity-value">${value}</div>
                </div>
            `;
        }
        
        perplexityStats.innerHTML = perplexityHtml;
    } catch (error) {
        perplexityStats.innerHTML = `
            <div style="grid-column: span 2; text-align: center; color: var(--error-color);">
                Failed to load model statistics. Please try again.
            </div>
        `;
    }
}

// Handle input with validation and debounce
const debouncedPrediction = debounce(() => getPrediction(false), 500);
contextInput.addEventListener('input', (e) => {
    const value = e.target.value;
    if (value && !validateInput(value)) {
        showError('Context can only contain letters and spaces.');
        hideResults();
        return;
    }
    debouncedPrediction();
});

// Event listeners
predictBtn.addEventListener('click', () => getPrediction(true));
clearBtn.addEventListener('click', clearInput);
usePredictBtn.addEventListener('click', usePrediction);
settingsBtn.addEventListener('click', openSettingsModal);
closeModal.addEventListener('click', closeSettingsModal);

// Handle click outside modal to close it
window.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        closeSettingsModal();
    }
});

// Handle ESC key to close modal
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && settingsModal.classList.contains('show')) {
        closeSettingsModal();
    }
});

// Handle Enter key
contextInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        getPrediction(true);
    }
}); 