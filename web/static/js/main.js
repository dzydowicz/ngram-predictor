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
    errorDiv.style.display = 'none';
    contextInput.focus();
}

// Hide prediction results
function hideResults() {
    resultDiv.classList.remove('show');
    predictionSpan.textContent = '';
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
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    resultDiv.classList.remove('show');
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
    errorDiv.style.display = 'none';
    
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
            errorDiv.style.display = 'none';
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

// Handle Enter key
contextInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        getPrediction(true);
    }
}); 