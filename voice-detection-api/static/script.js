// Configuration
const API_BASE_URL = window.location.origin;
const API_ENDPOINT = `${API_BASE_URL}/predict`;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const apiKeyInput = document.getElementById('apiKey');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadSection = document.querySelector('.upload-section');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const analyzeAnotherBtn = document.getElementById('analyzeAnother');
const tryAgainBtn = document.getElementById('tryAgain');

// State
let selectedFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIStatus();
    loadSavedAPIKey();
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        audioFileInput.click();
    });

    // File input change
    audioFileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Remove file
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // API key input
    apiKeyInput.addEventListener('input', () => {
        updateAnalyzeButton();
        saveAPIKey();
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeAudio);

    // Analyze another
    analyzeAnotherBtn.addEventListener('click', resetToUpload);

    // Try again
    tryAgainBtn.addEventListener('click', resetToUpload);
}

// File handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
        processFile(file);
    } else {
        showError('Please drop a valid audio file');
    }
}

function processFile(file) {
    selectedFile = file;
    
    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    uploadArea.style.display = 'none';
    fileInfo.style.display = 'flex';
    
    updateAnalyzeButton();
}

function clearFile() {
    selectedFile = null;
    audioFileInput.value = '';
    
    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    const hasFile = selectedFile !== null;
    const hasApiKey = apiKeyInput.value.trim() !== '';
    
    analyzeBtn.disabled = !(hasFile && hasApiKey);
}

// API Key management
function saveAPIKey() {
    const apiKey = apiKeyInput.value.trim();
    if (apiKey) {
        localStorage.setItem('voiceDetectionAPIKey', apiKey);
    }
}

function loadSavedAPIKey() {
    const savedKey = localStorage.getItem('voiceDetectionAPIKey');
    if (savedKey) {
        apiKeyInput.value = savedKey;
        updateAnalyzeButton();
    }
}

// API Status check
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'ok') {
            updateAPIStatus(true);
        } else {
            updateAPIStatus(false);
        }
    } catch (error) {
        updateAPIStatus(false);
    }
}

function updateAPIStatus(isOnline) {
    const statusIndicator = document.getElementById('apiStatus');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');
    
    if (isOnline) {
        statusDot.style.background = 'var(--accent-green)';
        statusText.textContent = 'API Ready';
        statusIndicator.style.background = 'rgba(81, 207, 102, 0.1)';
        statusIndicator.style.borderColor = 'rgba(81, 207, 102, 0.2)';
        statusText.style.color = 'var(--accent-green)';
    } else {
        statusDot.style.background = 'var(--accent-red)';
        statusText.textContent = 'API Offline';
        statusIndicator.style.background = 'rgba(255, 107, 107, 0.1)';
        statusIndicator.style.borderColor = 'rgba(255, 107, 107, 0.2)';
        statusText.style.color = 'var(--accent-red)';
    }
}

// Analyze audio
async function analyzeAudio() {
    if (!selectedFile || !apiKeyInput.value.trim()) {
        return;
    }

    // Show loading
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    try {
        // Convert file to base64
        const base64Audio = await fileToBase64(selectedFile);
        
        // Remove data URL prefix if present
        const base64Data = base64Audio.split(',')[1] || base64Audio;

        // Make API request
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKeyInput.value.trim()
            },
            body: JSON.stringify({
                audio_data: base64Data
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'Failed to analyze audio. Please check your API key and try again.');
    }
}

// Display results
function displayResults(data) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Classification
    const isAI = data.classification === 'AI_GENERATED';
    const resultBadge = document.getElementById('resultBadge');
    const resultClassification = document.getElementById('resultClassification');
    
    resultBadge.textContent = isAI ? '🤖 AI Detected' : '👤 Human Voice';
    resultBadge.className = `result-badge ${isAI ? 'ai' : 'human'}`;
    resultClassification.textContent = isAI ? 'AI-Generated Voice' : 'Human Voice';

    // Confidence
    const confidencePercent = (data.confidence_score * 100).toFixed(1);
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    
    const confidenceLevel = document.getElementById('confidenceLevel');
    confidenceLevel.textContent = data.confidence_level;
    confidenceLevel.className = `confidence-level ${data.confidence_level.toLowerCase()}`;

    // Metrics
    document.getElementById('languageValue').textContent = data.detected_language;
    
    const riskPercent = (data.deepfake_risk_score * 100).toFixed(1);
    document.getElementById('riskValue').textContent = `${riskPercent}%`;
    
    document.getElementById('qualityValue').textContent = data.audio_quality.quality_check;

    // Quality details
    document.getElementById('snrValue').textContent = `${data.audio_quality.snr.toFixed(2)} dB`;
    document.getElementById('clippingValue').textContent = data.audio_quality.clipping_detected ? 'Yes' : 'No';

    // Explainability
    const explainabilityList = document.getElementById('explainabilityList');
    explainabilityList.innerHTML = '';
    data.explainability.forEach(reason => {
        const li = document.createElement('li');
        li.textContent = reason;
        explainabilityList.appendChild(li);
    });

    // Segments
    if (data.segments && data.segments.length > 0) {
        const segmentsSection = document.getElementById('segmentsSection');
        const segmentsTimeline = document.getElementById('segmentsTimeline');
        
        segmentsSection.style.display = 'block';
        segmentsTimeline.innerHTML = '';
        
        data.segments.forEach(segment => {
            const segmentItem = document.createElement('div');
            segmentItem.className = 'segment-item';
            
            const isSegmentAI = segment.label === 'AI_GENERATED';
            const segmentConfidence = (segment.confidence * 100).toFixed(1);
            
            segmentItem.innerHTML = `
                <span class="segment-time">${segment.start_time.toFixed(2)}s - ${segment.end_time.toFixed(2)}s</span>
                <span class="segment-label ${isSegmentAI ? 'ai' : 'human'}">${isSegmentAI ? 'AI' : 'Human'}</span>
                <span class="segment-confidence">${segmentConfidence}%</span>
            `;
            
            segmentsTimeline.appendChild(segmentItem);
        });
    } else {
        document.getElementById('segmentsSection').style.display = 'none';
    }
}

// Error handling
function showError(message) {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

// Reset
function resetToUpload() {
    uploadSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    clearFile();
}

// Utility functions
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
