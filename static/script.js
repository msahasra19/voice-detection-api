// Voice Anti-Spoofing & Multilingual Forensics Client Logic

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const tabForensics = document.getElementById("tabForensics");
    const tabAblation = document.getElementById("tabAblation");
    const forensicsView = document.getElementById("forensicsView");
    const ablationView = document.getElementById("ablationView");

    const uploadArea = document.getElementById("uploadArea");
    const audioFileInput = document.getElementById("audioFileInput");
    const fileLoadedBanner = document.getElementById("fileLoadedBanner");
    const loadedFileName = document.getElementById("loadedFileName");
    const loadedFileSize = document.getElementById("loadedFileSize");
    const btnClearFile = document.getElementById("btnClearFile");

    const btnRecordMic = document.getElementById("btnRecordMic");
    const recordBtnText = document.getElementById("recordBtnText");
    const recordTimer = document.getElementById("recordTimer");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const loadingCard = document.getElementById("loadingCard");
    const resultsSection = document.getElementById("resultsSection");

    // Verdict Elements
    const verdictCard = document.getElementById("verdictCard");
    const verdictPill = document.getElementById("verdictPill");
    const verdictPillText = document.getElementById("verdictPillText");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictSummary = document.getElementById("verdictSummary");
    const verdictConfidence = document.getElementById("verdictConfidence");
    const verdictConfLevel = document.getElementById("verdictConfLevel");
    const verdictLatency = document.getElementById("verdictLatency");
    const verdictEngine = document.getElementById("verdictEngine");
    const riskGaugeCircle = document.getElementById("riskGaugeCircle");
    const riskGaugeValue = document.getElementById("riskGaugeValue");

    // Branch 1 Elements
    const mlProbValue = document.getElementById("mlProbValue");
    const mlProbBar = document.getElementById("mlProbBar");
    const mlFlatnessValue = document.getElementById("mlFlatnessValue");

    // Branch 2 Elements
    const acousticSnrValue = document.getElementById("acousticSnrValue");
    const pitchMeanStd = document.getElementById("pitchMeanStd");
    const pitchStabilityVal = document.getElementById("pitchStabilityVal");
    const silenceRatioVal = document.getElementById("silenceRatioVal");
    const qualityCheckVal = document.getElementById("qualityCheckVal");

    // Branch 3 Elements
    const detectedLangValue = document.getElementById("detectedLangValue");
    const langConfidenceVal = document.getElementById("langConfidenceVal");
    const transcriptText = document.getElementById("transcriptText");

    // Timeline & Explainability
    const timelineTrack = document.getElementById("timelineTrack");
    const segmentHoverDetail = document.getElementById("segmentHoverDetail");
    const hoverTimeSpan = document.getElementById("hoverTimeSpan");
    const hoverVerdict = document.getElementById("hoverVerdict");
    const hoverRisk = document.getElementById("hoverRisk");
    const hoverSnr = document.getElementById("hoverSnr");
    const explainabilityList = document.getElementById("explainabilityList");

    const ablationTableBody = document.getElementById("ablationTableBody");
    const btnRefreshAblation = document.getElementById("btnRefreshAblation");

    // State Variables
    let currentAudioPayload = null; // base64 string or File
    let mediaRecorder = null;
    let audioChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;

    // Tab Navigation
    tabForensics.addEventListener("click", () => {
        tabForensics.classList.add("active");
        tabAblation.classList.remove("active");
        forensicsView.style.display = "block";
        ablationView.style.display = "none";
    });

    tabAblation.addEventListener("click", () => {
        tabAblation.classList.add("active");
        tabForensics.classList.remove("active");
        forensicsView.style.display = "none";
        ablationView.style.display = "block";
        loadAblationReport();
    });

    // File Upload Handlers
    uploadArea.addEventListener("click", (e) => {
        if (e.target !== btnClearFile && !btnClearFile.contains(e.target)) {
            audioFileInput.click();
        }
    });

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "var(--color-cyan)";
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.style.borderColor = "";
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnClearFile.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSelectedAudio();
    });

    function handleFileSelect(file) {
        currentAudioPayload = file;
        loadedFileName.textContent = file.name;
        loadedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
        fileLoadedBanner.style.display = "flex";
        btnAnalyze.disabled = false;
    }

    function clearSelectedAudio() {
        currentAudioPayload = null;
        audioFileInput.value = "";
        fileLoadedBanner.style.display = "none";
        btnAnalyze.disabled = true;
    }

    // Microphone Recording
    btnRecordMic.addEventListener("click", async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            btnRecordMic.classList.remove("recording");
            recordBtnText.textContent = "Record from Microphone";
            clearInterval(recordInterval);
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    try {
                        const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
                        const arrayBuffer = await rawBlob.arrayBuffer();
                        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                        
                        // Extract mono PCM samples
                        let channelData;
                        if (audioBuffer.numberOfChannels > 1) {
                            const ch0 = audioBuffer.getChannelData(0);
                            const ch1 = audioBuffer.getChannelData(1);
                            channelData = new Float32Array(ch0.length);
                            for (let i = 0; i < ch0.length; i++) {
                                channelData[i] = (ch0[i] + ch1[i]) / 2.0;
                            }
                        } else {
                            channelData = audioBuffer.getChannelData(0);
                        }
                        
                        createPresetFile(channelData, audioBuffer.sampleRate, "microphone_recording.wav");
                        audioCtx.close();
                    } catch (decErr) {
                        // Fallback: send raw blob
                        const rawBlob = new Blob(audioChunks, { type: "audio/webm" });
                        const file = new File([rawBlob], "microphone_recording.webm", { type: "audio/webm" });
                        handleFileSelect(file);
                    }
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                btnRecordMic.classList.add("recording");
                recordBtnText.textContent = "Stop Recording";
                recordSeconds = 0;
                recordTimer.textContent = "00:00";
                recordInterval = setInterval(() => {
                    recordSeconds++;
                    const m = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
                    const s = String(recordSeconds % 60).padStart(2, "0");
                    recordTimer.textContent = `${m}:${s}`;
                }, 1000);
            } catch (err) {
                alert("Microphone access failed: " + err.message);
            }
        }
    });

    // Preset Audio Generators (for instant verification)
    document.querySelectorAll(".btn-preset").forEach(btn => {
        btn.addEventListener("click", () => {
            const preset = btn.dataset.preset;
            generatePresetSample(preset);
        });
    });

    function generatePresetSample(presetType) {
        const sr = 16000;
        const duration = 3.0;
        const totalSamples = sr * duration;
        const buffer = new Float32Array(totalSamples);

        if (presetType === "human_clean") {
            // Human dynamic pitch & formants + breathing envelope
            for (let i = 0; i < totalSamples; i++) {
                const t = i / sr;
                const f0 = 140 + 35 * Math.sin(2 * Math.PI * 1.8 * t);
                const env = Math.pow(Math.sin(2 * Math.PI * 3.0 * t), 2);
                buffer[i] = (0.6 * Math.sin(2 * Math.PI * f0 * t) + 0.3 * Math.sin(2 * Math.PI * 2 * f0 * t)) * env + (Math.random() - 0.5) * 0.04;
            }
            createPresetFile(buffer, sr, "human_clean_sample.wav");
        } else if (presetType === "ai_synth") {
            // Pristine zero-noise flat pitch TTS
            const f0 = 165.0;
            for (let i = 0; i < totalSamples; i++) {
                const t = i / sr;
                buffer[i] = (0.7 * Math.sin(2 * Math.PI * f0 * t) + 0.3 * Math.sin(2 * Math.PI * 2 * f0 * t)) * 0.9;
            }
            createPresetFile(buffer, sr, "ai_synthetic_tts.wav");
        } else if (presetType === "ai_vocoder") {
            // Robotic vocoder artifact & high spectral flatness
            for (let i = 0; i < totalSamples; i++) {
                const t = i / sr;
                buffer[i] = Math.sin(2 * Math.PI * 180 * t) + 0.35 * Math.sin(2 * Math.PI * 3200 * t) + (Math.random() - 0.5) * 0.15;
            }
            createPresetFile(buffer, sr, "ai_neural_vocoder.wav");
        } else {
            // Multilingual speech with syllables
            for (let i = 0; i < totalSamples; i++) {
                const t = i / sr;
                const f0 = 190 + 40 * Math.sin(2 * Math.PI * 2.5 * t);
                const syllable = Math.sin(2 * Math.PI * 4 * t) > 0 ? 1 : 0.1;
                buffer[i] = Math.sin(2 * Math.PI * f0 * t) * syllable + (Math.random() - 0.5) * 0.03;
            }
            createPresetFile(buffer, sr, "multilingual_human_sample.wav");
        }
    }

    function createPresetFile(floatArray, sampleRate, filename) {
        const wavBytes = encodeWAV(floatArray, sampleRate);
        const blob = new Blob([wavBytes], { type: "audio/wav" });
        const file = new File([blob], filename, { type: "audio/wav" });
        handleFileSelect(file);
    }

    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }
        return new Uint8Array(buffer);
    }

    // Run Forensics Analysis
    btnAnalyze.addEventListener("click", async () => {
        if (!currentAudioPayload) return;

        btnAnalyze.disabled = true;
        loadingCard.style.display = "block";
        resultsSection.style.display = "none";

        try {
            let response;
            if (currentAudioPayload instanceof File) {
                const formData = new FormData();
                formData.append("file", currentAudioPayload);
                response = await fetch("/predict-file", {
                    method: "POST",
                    body: formData
                });
            } else {
                response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ audio_base64: currentAudioPayload })
                });
            }

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Analysis request failed.");
            }

            const data = await response.json();
            renderResults(data);
        } catch (err) {
            alert("Forensics Analysis Error: " + err.message);
        } finally {
            loadingCard.style.display = "none";
            btnAnalyze.disabled = false;
        }
    });

    function renderResults(data) {
        resultsSection.style.display = "flex";

        const isAI = data.classification === "AI_GENERATED";
        const riskPct = Math.round(data.deepfake_risk_score * 100);
        const confPct = Math.round(data.confidence_score * 100);

        // Verdict Card Styling
        verdictCard.className = `card verdict-card ${isAI ? 'ai-verdict' : 'human-verdict'}`;
        verdictPill.className = `verdict-pill ${isAI ? 'ai-pill' : 'human-pill'}`;
        verdictPillText.textContent = isAI ? "AI GENERATED (SPOOFED)" : "BONA FIDE HUMAN";

        verdictTitle.textContent = isAI ? "Synthetic / AI Voice Detected" : "Authentic Human Speech";
        verdictSummary.textContent = isAI
            ? "Acoustic spectral patterns, pitch stability, and signal dynamics indicate AI speech synthesis or voice conversion."
            : "Acoustic formants, pitch variation, and natural temporal pauses confirm genuine human speech.";

        verdictConfidence.textContent = `${confPct}%`;
        verdictConfLevel.textContent = data.confidence_level;
        verdictLatency.textContent = `${data.processing_time_ms} ms`;
        verdictEngine.textContent = data.fusion_analysis.fusion_method === "learned_logistic_regression"
            ? "Learned Fusion (RF + Logistic Regression)"
            : "Calibrated Acoustic Rule Fusion";

        riskGaugeValue.textContent = `${riskPct}%`;
        riskGaugeCircle.className = `gauge-circle ${riskPct >= 50 ? 'risk-high' : 'risk-low'}`;

        // Branch 1: ML
        const mlPct = Math.round(data.fusion_analysis.ml_probability * 100);
        mlProbValue.textContent = `${mlPct}%`;
        mlProbBar.style.width = `${mlPct}%`;
        mlFlatnessValue.textContent = data.acoustic_indicators.spectral_flatness_mean.toFixed(4);

        // Branch 2: Acoustics
        acousticSnrValue.textContent = `${data.acoustic_indicators.snr_db} dB`;
        pitchMeanStd.textContent = `${data.acoustic_indicators.pitch_mean_hz} Hz (±${data.acoustic_indicators.pitch_std_hz} Hz)`;
        pitchStabilityVal.textContent = `${data.acoustic_indicators.pitch_stability_score} (${data.acoustic_indicators.pitch_stability_score > 0.5 ? 'Monotone/AI' : 'Natural'})`;
        silenceRatioVal.textContent = `${(data.acoustic_indicators.silence_ratio * 100).toFixed(1)}%`;
        qualityCheckVal.textContent = data.acoustic_indicators.quality_check;

        // Branch 3: Whisper Language
        detectedLangValue.textContent = data.detected_language;
        langConfidenceVal.textContent = `${Math.round(data.language_confidence * 100)}%`;
        transcriptText.textContent = data.sample_transcript ? `"${data.sample_transcript}"` : "(No clear speech transcribed in snippet)";

        // 1-Second Window Timeline
        timelineTrack.innerHTML = "";
        segmentHoverDetail.style.display = "none";

        if (data.segments && data.segments.length > 0) {
            data.segments.forEach((seg, idx) => {
                const segDiv = document.createElement("div");
                const isSegAI = seg.label === "AI_GENERATED";
                segDiv.className = `timeline-seg-block ${isSegAI ? 'seg-ai' : 'seg-safe'}`;
                segDiv.textContent = `${Math.floor(seg.start_time)}s`;

                segDiv.addEventListener("mouseenter", () => {
                    segmentHoverDetail.style.display = "block";
                    hoverTimeSpan.textContent = `${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s`;
                    hoverVerdict.textContent = isSegAI ? "Suspicious / AI" : "Bona Fide Human";
                    hoverRisk.textContent = `${Math.round(seg.risk_score * 100)}%`;
                    hoverSnr.textContent = seg.snr_db ? `${seg.snr_db} dB` : "N/A";
                });

                timelineTrack.appendChild(segDiv);
            });
        }

        // Explainability
        explainabilityList.innerHTML = "";
        data.explainability.forEach(note => {
            const li = document.createElement("li");
            li.textContent = note;
            explainabilityList.appendChild(li);
        });

        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    // Ablation Report Loader
    async function loadAblationReport() {
        ablationTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px;">Loading ablation experimental metrics...</td></tr>`;
        try {
            const res = await fetch("/ablation-report");
            const data = await res.json();
            ablationTableBody.innerHTML = "";

            data.configurations.forEach(cfg => {
                const tr = document.createElement("tr");
                const isProposed = cfg.config_id.includes("D");
                tr.innerHTML = `
                    <td><span class="config-badge ${isProposed ? 'badge-proposed' : ''}">${cfg.config_id}</span></td>
                    <td><b>${cfg.name}</b><br><small style="color:var(--text-muted)">${cfg.features}</small></td>
                    <td>${cfg.classifier}</td>
                    <td><b>${(cfg.accuracy * 100).toFixed(2)}%</b></td>
                    <td>${(cfg.roc_auc).toFixed(4)}</td>
                    <td><b style="color:${cfg.eer < 0.05 ? 'var(--color-emerald)' : 'var(--color-amber)'}">${(cfg.eer * 100).toFixed(2)}%</b></td>
                    <td><span class="status-badge" style="display:inline-flex;">Validated</span></td>
                `;
                ablationTableBody.appendChild(tr);
            });
        } catch (e) {
            ablationTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--color-rose);">Failed to load ablation report.</td></tr>`;
        }
    }

    btnRefreshAblation.addEventListener("click", loadAblationReport);
});
