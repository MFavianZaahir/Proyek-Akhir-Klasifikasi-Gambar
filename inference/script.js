let model;
const classNames = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'];

async function loadModel() {
    try {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('loading').innerHTML = "Creating model architecture...";
        
        model = await createModel();
        
        document.getElementById('loading').innerHTML = "Loading pre-trained weights...";
        try {
            await attemptLoadWeights(model);
            console.log("Model ready with pre-trained weights");
        } catch (error) {
            console.warn("Using model without pre-trained weights");
        }
        
        document.getElementById('loading').style.display = 'none';
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        document.getElementById('loading').innerHTML = 
            `Error loading model: ${error.message}<br>
            Check console for details.`;
        throw error;
    }
}

async function predictImage(imageElement) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result-container').style.display = 'none';
    
    try {
        if (!model) {
            await loadModel();
        }
        
        const tensor = tf.tidy(() => {
            const imgTensor = tf.browser.fromPixels(imageElement);
            const resized = tf.image.resizeBilinear(imgTensor, [228, 228]);
            const normalized = resized.toFloat().div(255.0);
            return normalized.expandDims(0);
        });
        
        const predictions = await model.predict(tensor).data();
        displayResults(predictions);
        tensor.dispose();
        
    } catch (error) {
        console.error('Error during prediction:', error);
        document.getElementById('predictions').innerHTML = 
            'Error: Could not process the image. ' + error.message;
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('result-container').style.display = 'block';
    }
}

function displayResults(predictions) {
    const predictionContainer = document.getElementById('predictions');
    predictionContainer.innerHTML = '';
    
    const indexedPredictions = Array.from(predictions).map((prob, i) => ({prob, index: i}));
    indexedPredictions.sort((a, b) => b.prob - a.prob);
    
    for (let i = 0; i < indexedPredictions.length; i++) {
        const prediction = indexedPredictions[i];
        const label = classNames[prediction.index];
        const probability = (prediction.prob * 100).toFixed(2);
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'prediction-label';
        labelDiv.innerHTML = `<span>${label}</span><span>${probability}%</span>`;
        predictionContainer.appendChild(labelDiv);
        
        const barContainer = document.createElement('div');
        barContainer.style.width = '100%';
        barContainer.style.backgroundColor = '#e9ecef';
        barContainer.style.marginBottom = '10px';
        barContainer.style.borderRadius = '3px';
        barContainer.style.overflow = 'hidden';
        
        const bar = document.createElement('div');
        bar.className = 'prediction-bar';
        bar.style.width = '0%';
        setTimeout(() => { bar.style.width = `${probability}%`; }, 10);
        
        barContainer.appendChild(bar);
        predictionContainer.appendChild(barContainer);
    }
}

document.getElementById('file-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                const canvas = document.getElementById('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                predictImage(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

document.addEventListener('DOMContentLoaded', function() {
    loadModel();
});