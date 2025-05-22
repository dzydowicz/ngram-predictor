async function getPrediction() {
    const contextInput = document.getElementById('context');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const predictionSpan = document.getElementById('prediction');
    
    const context = contextInput.value.trim();
    const words = context.split(/\s+/);
    
    // Clear previous results
    resultDiv.classList.remove('show');
    errorDiv.style.display = 'none';
    
    if (words.length > 3) {
        errorDiv.textContent = 'Error: Maximum context length is 3 words.';
        errorDiv.style.display = 'block';
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ context: words })
        });
        
        const data = await response.json();
        
        if (data.error) {
            errorDiv.textContent = data.error;
            errorDiv.style.display = 'block';
        } else {
            predictionSpan.textContent = data.prediction;
            resultDiv.classList.add('show');
        }
    } catch (error) {
        errorDiv.textContent = 'Error: Could not connect to the server.';
        errorDiv.style.display = 'block';
    }
} 