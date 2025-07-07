// Neural Network Visualization Script
class NeuralNetworkVisualizer {
    constructor() {
        this.canvas = document.getElementById('inputCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.svg = document.getElementById('networkSvg');
        this.isDrawing = false;
        this.isPredicting = false;
        
        // Network structure
        this.layers = [
            { name: 'Input', neurons: 784, x: 50, y: 50, displayCount: 16 },
            { name: 'Conv1', neurons: 32, x: 250, y: 50, displayCount: 8 },
            { name: 'Conv2', neurons: 64, x: 450, y: 50, displayCount: 10 },
            { name: 'Output', neurons: 10, x: 650, y: 50, displayCount: 10 }
        ];
        
        // Sample dataset for demo
        this.sampleDigits = {
            '0': this.createSampleDigit0(),
            '1': this.createSampleDigit1(),
            '2': this.createSampleDigit2(),
            '3': this.createSampleDigit3(),
            '4': this.createSampleDigit4(),
            '5': this.createSampleDigit5(),
            '6': this.createSampleDigit6(),
            '7': this.createSampleDigit7(),
            '8': this.createSampleDigit8(),
            '9': this.createSampleDigit9()
        };
        
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.setupEventListeners();
        this.drawNetwork();
        this.resetPredictions();
    }
    
    setupCanvas() {
        // Set canvas size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = 280;
        this.canvas.height = 280;
        
        // Style canvas
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 8;
        
        // Clear canvas
        this.clearCanvas();
    }
    
    setupEventListeners() {
        // Canvas drawing events
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
        
        // Button events
        document.getElementById('clearBtn').addEventListener('click', this.clearCanvas.bind(this));
        document.getElementById('predictBtn').addEventListener('click', this.predict.bind(this));
        document.getElementById('digitSelect').addEventListener('change', this.loadSampleDigit.bind(this));
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.canvas.classList.add('drawing');
        this.draw(e);
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        
        // Auto-predict while drawing (throttled)
        this.throttledPredict();
    }
    
    stopDrawing() {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        this.canvas.classList.remove('drawing');
        this.ctx.beginPath();
        
        // Final prediction after drawing stops
        setTimeout(() => this.predict(), 300);
    }
    
    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                        e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.canvas.dispatchEvent(mouseEvent);
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.beginPath();
        this.resetPredictions();
        this.resetNetworkVisualization();
        document.getElementById('digitSelect').value = '';
    }
    
    loadSampleDigit() {
        const select = document.getElementById('digitSelect');
        const digit = select.value;
        
        if (digit && this.sampleDigits[digit]) {
            this.clearCanvas();
            this.drawSampleDigit(this.sampleDigits[digit]);
            setTimeout(() => this.predict(), 500);
        }
    }
    
    drawSampleDigit(digitData) {
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 8;
        
        digitData.forEach(path => {
            this.ctx.beginPath();
            path.forEach((point, index) => {
                if (index === 0) {
                    this.ctx.moveTo(point.x, point.y);
                } else {
                    this.ctx.lineTo(point.x, point.y);
                }
            });
            this.ctx.stroke();
        });
    }
    
    throttledPredict = this.throttle(() => {
        if (!this.isPredicting) {
            this.predict();
        }
    }, 500);
    
    throttle(func, delay) {
        let timeoutId;
        let lastExecTime = 0;
        return function (...args) {
            const currentTime = Date.now();
            
            if (currentTime - lastExecTime > delay) {
                func.apply(this, args);
                lastExecTime = currentTime;
            } else {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    func.apply(this, args);
                    lastExecTime = Date.now();
                }, delay - (currentTime - lastExecTime));
            }
        };
    }
    
    async predict() {
        if (this.isPredicting) return;
        
        this.isPredicting = true;
        const predictBtn = document.getElementById('predictBtn');
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="loading"></span> İşleniyor...';
        
        try {
            // Get canvas data
            const imageData = this.getCanvasImageData();
            
            // Simulate network processing or make actual API call
            const predictions = await this.makePrediction(imageData);
            
            // Update UI with results
            this.updatePredictionBars(predictions.probabilities);
            this.updatePredictionResult(predictions.prediction, predictions.confidence);
            this.animateNetwork(predictions.activations);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError('Tahmin sırasında hata oluştu');
        } finally {
            this.isPredicting = false;
            predictBtn.disabled = false;
            predictBtn.innerHTML = 'Tahmin Et';
        }
    }
    
    getCanvasImageData() {
        // Convert canvas to 28x28 grayscale image data
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Resize to 28x28 and convert to grayscale
        const resized = [];
        const scaleX = this.canvas.width / 28;
        const scaleY = this.canvas.height / 28;
        
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const sourceX = Math.floor(x * scaleX);
                const sourceY = Math.floor(y * scaleY);
                const index = (sourceY * this.canvas.width + sourceX) * 4;
                
                // Convert RGBA to grayscale (invert for MNIST format)
                const gray = 255 - (data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114);
                resized.push(gray / 255.0);
            }
        }
        
        return resized;
    }
    
    async makePrediction(imageData) {
        // Try to make actual API call to the model service
        try {
            const response = await fetch('http://localhost:8001/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image_data: imageData })
            });
            
            if (response.ok) {
                const result = await response.json();
                return {
                    prediction: result.prediction,
                    confidence: Math.round(result.confidence * 100),
                    probabilities: result.probabilities || this.generateMockProbabilities(result.prediction),
                    activations: this.generateMockActivations()
                };
            }
        } catch (error) {
            console.log('API not available, using mock prediction');
        }
        
        // Fallback to mock prediction for demo
        return this.generateMockPrediction();
    }
    
    generateMockPrediction() {
        // Generate realistic-looking mock prediction
        const prediction = Math.floor(Math.random() * 10);
        const confidence = 75 + Math.random() * 20; // 75-95%
        const probabilities = Array(10).fill(0).map(() => Math.random() * 0.1);
        probabilities[prediction] = confidence / 100;
        
        // Normalize probabilities
        const sum = probabilities.reduce((a, b) => a + b, 0);
        const normalizedProbs = probabilities.map(p => p / sum);
        
        return {
            prediction,
            confidence: Math.round(confidence),
            probabilities: normalizedProbs,
            activations: this.generateMockActivations()
        };
    }
    
    generateMockProbabilities(prediction) {
        const probs = Array(10).fill(0).map(() => Math.random() * 0.1);
        probs[prediction] = 0.7 + Math.random() * 0.25;
        const sum = probs.reduce((a, b) => a + b, 0);
        return probs.map(p => p / sum);
    }
    
    generateMockActivations() {
        return this.layers.map(layer => 
            Array(layer.displayCount).fill(0).map(() => Math.random())
        );
    }
    
    updatePredictionBars(probabilities) {
        probabilities.forEach((prob, index) => {
            const container = document.querySelector(`[data-digit="${index}"]`);
            const fill = container.querySelector('.bar-fill');
            const percentage = container.querySelector('.percentage');
            
            // Animate bar fill
            setTimeout(() => {
                fill.style.width = `${prob * 100}%`;
                percentage.textContent = `${Math.round(prob * 100)}%`;
                
                // Highlight highest probability
                if (prob === Math.max(...probabilities)) {
                    container.classList.add('highlight');
                    setTimeout(() => container.classList.remove('highlight'), 2000);
                }
            }, index * 50); // Stagger animations
        });
    }
    
    updatePredictionResult(prediction, confidence) {
        document.getElementById('predictedDigit').textContent = prediction;
        document.getElementById('confidence').textContent = `${confidence}%`;
    }
    
    resetPredictions() {
        // Reset all prediction bars
        document.querySelectorAll('.bar-fill').forEach(fill => {
            fill.style.width = '0%';
        });
        document.querySelectorAll('.percentage').forEach(perc => {
            perc.textContent = '0%';
        });
        document.querySelectorAll('.bar-container').forEach(container => {
            container.classList.remove('highlight');
        });
        
        // Reset prediction result
        document.getElementById('predictedDigit').textContent = '-';
        document.getElementById('confidence').textContent = '-%';
    }
    
    drawNetwork() {
        this.drawConnections();
        this.drawNeurons();
    }
    
    drawConnections() {
        const connectionsGroup = document.getElementById('connections');
        connectionsGroup.innerHTML = '';
        
        // Draw connections between layers
        for (let i = 0; i < this.layers.length - 1; i++) {
            const currentLayer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            for (let j = 0; j < currentLayer.displayCount; j++) {
                for (let k = 0; k < nextLayer.displayCount; k++) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    
                    const x1 = currentLayer.x + 40;
                    const y1 = currentLayer.y + (j * 35) + 20;
                    const x2 = nextLayer.x - 10;
                    const y2 = nextLayer.y + (k * 35) + 20;
                    
                    line.setAttribute('x1', x1);
                    line.setAttribute('y1', y1);
                    line.setAttribute('x2', x2);
                    line.setAttribute('y2', y2);
                    line.setAttribute('class', 'connection');
                    line.setAttribute('data-from', `${i}-${j}`);
                    line.setAttribute('data-to', `${i + 1}-${k}`);
                    
                    connectionsGroup.appendChild(line);
                }
            }
        }
    }
    
    drawNeurons() {
        const nodesGroup = document.getElementById('nodes');
        nodesGroup.innerHTML = '';
        
        this.layers.forEach((layer, layerIndex) => {
            for (let i = 0; i < layer.displayCount; i++) {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', layer.x);
                circle.setAttribute('cy', layer.y + (i * 35) + 20);
                circle.setAttribute('r', 12);
                circle.setAttribute('fill', '#e2e8f0');
                circle.setAttribute('stroke', '#cbd5e1');
                circle.setAttribute('stroke-width', 2);
                circle.setAttribute('class', 'neuron');
                circle.setAttribute('data-layer', layerIndex);
                circle.setAttribute('data-neuron', i);
                
                nodesGroup.appendChild(circle);
            }
        });
    }
    
    animateNetwork(activations) {
        if (!activations) return;
        
        // Reset previous animations
        this.resetNetworkVisualization();
        
        // Animate layer by layer with delays
        activations.forEach((layerActivations, layerIndex) => {
            setTimeout(() => {
                this.animateLayer(layerIndex, layerActivations);
                if (layerIndex < activations.length - 1) {
                    this.animateConnections(layerIndex, layerActivations, activations[layerIndex + 1]);
                }
            }, layerIndex * 800);
        });
    }
    
    animateLayer(layerIndex, activations) {
        activations.forEach((activation, neuronIndex) => {
            const neuron = document.querySelector(`[data-layer="${layerIndex}"][data-neuron="${neuronIndex}"]`);
            if (neuron) {
                const intensity = Math.max(0.2, activation);
                const color = this.getActivationColor(intensity);
                
                neuron.setAttribute('fill', color);
                neuron.classList.add('active');
                
                // Remove active class after animation
                setTimeout(() => {
                    neuron.classList.remove('active');
                }, 1500);
            }
        });
    }
    
    animateConnections(fromLayer, fromActivations, toActivations) {
        setTimeout(() => {
            fromActivations.forEach((fromActivation, fromIndex) => {
                toActivations.forEach((toActivation, toIndex) => {
                    const connection = document.querySelector(
                        `[data-from="${fromLayer}-${fromIndex}"][data-to="${fromLayer + 1}-${toIndex}"]`
                    );
                    
                    if (connection && fromActivation > 0.3 && toActivation > 0.3) {
                        connection.classList.add('active');
                        connection.setAttribute('stroke-dasharray', '5,5');
                        
                        setTimeout(() => {
                            connection.classList.remove('active');
                            connection.removeAttribute('stroke-dasharray');
                        }, 2000);
                    }
                });
            });
        }, 400);
    }
    
    getActivationColor(intensity) {
        // Create color gradient based on activation intensity
        const colors = [
            '#e2e8f0', // Very low
            '#bfdbfe', // Low
            '#60a5fa', // Medium
            '#3b82f6', // High
            '#1d4ed8'  // Very high
        ];
        
        const index = Math.min(Math.floor(intensity * colors.length), colors.length - 1);
        return colors[index];
    }
    
    resetNetworkVisualization() {
        // Reset all neurons
        document.querySelectorAll('.neuron').forEach(neuron => {
            neuron.setAttribute('fill', '#e2e8f0');
            neuron.classList.remove('active');
        });
        
        // Reset all connections
        document.querySelectorAll('.connection').forEach(connection => {
            connection.classList.remove('active');
            connection.removeAttribute('stroke-dasharray');
        });
    }
    
    showError(message) {
        // Simple error display
        const result = document.getElementById('predictionResult');
        result.style.background = 'rgba(239, 68, 68, 0.1)';
        result.style.borderColor = 'rgba(239, 68, 68, 0.2)';
        document.getElementById('predictedDigit').textContent = 'Hata';
        document.getElementById('confidence').textContent = message;
        
        setTimeout(() => {
            result.style.background = '';
            result.style.borderColor = '';
            this.resetPredictions();
        }, 3000);
    }
    
    // Sample digit drawing functions
    createSampleDigit0() {
        return [[
            {x: 100, y: 80}, {x: 120, y: 70}, {x: 150, y: 70}, {x: 170, y: 80},
            {x: 180, y: 100}, {x: 180, y: 150}, {x: 170, y: 180}, {x: 150, y: 190},
            {x: 120, y: 190}, {x: 100, y: 180}, {x: 90, y: 150}, {x: 90, y: 100}, {x: 100, y: 80}
        ]];
    }
    
    createSampleDigit1() {
        return [
            [{x: 120, y: 80}, {x: 140, y: 70}],
            [{x: 140, y: 70}, {x: 140, y: 190}],
            [{x: 120, y: 190}, {x: 160, y: 190}]
        ];
    }
    
    createSampleDigit2() {
        return [[
            {x: 100, y: 90}, {x: 120, y: 70}, {x: 150, y: 70}, {x: 170, y: 90},
            {x: 170, y: 110}, {x: 100, y: 170}, {x: 180, y: 170}, {x: 180, y: 190},
            {x: 90, y: 190}
        ]];
    }
    
    createSampleDigit3() {
        return [
            [
                {x: 100, y: 80}, {x: 120, y: 70}, {x: 150, y: 70}, {x: 170, y: 80},
                {x: 180, y: 100}, {x: 170, y: 120}, {x: 150, y: 130}
            ],
            [
                {x: 150, y: 130}, {x: 170, y: 140}, {x: 180, y: 160}, {x: 170, y: 180},
                {x: 150, y: 190}, {x: 120, y: 190}, {x: 100, y: 180}
            ]
        ];
    }
    
    createSampleDigit4() {
        return [
            [{x: 140, y: 70}, {x: 140, y: 190}],
            [{x: 120, y: 70}, {x: 120, y: 130}, {x: 180, y: 130}],
            [{x: 180, y: 110}, {x: 180, y: 190}]
        ];
    }
    
    createSampleDigit5() {
        return [[
            {x: 180, y: 70}, {x: 100, y: 70}, {x: 100, y: 120}, {x: 150, y: 120},
            {x: 170, y: 130}, {x: 180, y: 150}, {x: 170, y: 180}, {x: 150, y: 190},
            {x: 120, y: 190}, {x: 100, y: 180}
        ]];
    }
    
    createSampleDigit6() {
        return [[
            {x: 170, y: 80}, {x: 150, y: 70}, {x: 120, y: 70}, {x: 100, y: 80},
            {x: 90, y: 100}, {x: 90, y: 130}, {x: 100, y: 140}, {x: 150, y: 140},
            {x: 170, y: 150}, {x: 180, y: 170}, {x: 170, y: 180}, {x: 150, y: 190},
            {x: 120, y: 190}, {x: 100, y: 180}, {x: 90, y: 160}
        ]];
    }
    
    createSampleDigit7() {
        return [
            [{x: 100, y: 70}, {x: 180, y: 70}, {x: 140, y: 190}]
        ];
    }
    
    createSampleDigit8() {
        return [
            [
                {x: 140, y: 70}, {x: 120, y: 70}, {x: 100, y: 80}, {x: 100, y: 100},
                {x: 110, y: 110}, {x: 130, y: 120}, {x: 150, y: 120}, {x: 170, y: 110},
                {x: 180, y: 100}, {x: 180, y: 80}, {x: 160, y: 70}, {x: 140, y: 70}
            ],
            [
                {x: 130, y: 120}, {x: 110, y: 130}, {x: 100, y: 150}, {x: 100, y: 170},
                {x: 110, y: 180}, {x: 130, y: 190}, {x: 150, y: 190}, {x: 170, y: 180},
                {x: 180, y: 170}, {x: 180, y: 150}, {x: 170, y: 130}, {x: 150, y: 120}
            ]
        ];
    }
    
    createSampleDigit9() {
        return [[
            {x: 100, y: 180}, {x: 120, y: 190}, {x: 150, y: 190}, {x: 170, y: 180},
            {x: 180, y: 160}, {x: 180, y: 130}, {x: 170, y: 120}, {x: 120, y: 120},
            {x: 100, y: 110}, {x: 90, y: 90}, {x: 100, y: 80}, {x: 120, y: 70},
            {x: 150, y: 70}, {x: 170, y: 80}, {x: 180, y: 100}, {x: 180, y: 190}
        ]];
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new NeuralNetworkVisualizer();
});

// Add some global utility functions
window.toggleDarkMode = function() {
    document.documentElement.classList.toggle('dark-mode');
};

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch(e.key) {
            case 'Enter':
                e.preventDefault();
                document.getElementById('predictBtn').click();
                break;
            case 'Backspace':
                e.preventDefault();
                document.getElementById('clearBtn').click();
                break;
        }
    }
});
