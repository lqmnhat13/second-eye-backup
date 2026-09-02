/**
 * Second Eye - Frontend Engine
 * Dual-Mode System:
 *   1. Obstacle Detection & 2D Spatial Radar Mode
 *   2. OCR Document & Label Reader Mode with Native Vietnamese Voice
 */

document.addEventListener("DOMContentLoaded", () => {
    // SYSTEM STATE
    const state = {
        mode: "nav", // "nav" (Obstacle/Radar) or "ocr" (Document Reader)
        isStreaming: false,
        facingMode: "environment", // "user" or "environment" (rear camera)
        focalLength: 650,
        confidence: 0.35,
        voiceEnabled: true,
        speechRate: 1.1,
        dangerThreshold: 1.0,
        warningThreshold: 2.0,
        isProcessingFrame: false,
        isReadingOCR: false,
        disabledClasses: new Set(),
        currentObjects: [],
        lastSpokenPhrase: "",
        lastAlertTime: 0,
        radarAngle: 0,
        radarObjects: [],
        
        // OCR State
        ocrResult: null,
        ocrParagraphs: [],
        ocrCurrentParaIdx: 0,
        ocrIsPlaying: false,
        ocrFontSize: 1.15
    };

    // DOM ELEMENTS - CORE
    const video = document.getElementById("webcam-video");
    const hudCanvas = document.getElementById("hud-canvas");
    const hudCtx = hudCanvas.getContext("2d");
    const radarCanvas = document.getElementById("radar-canvas");
    const radarCtx = radarCanvas.getContext("2d");

    // Mode Switcher Tabs
    const tabModeNav = document.getElementById("tab-mode-nav");
    const tabModeOcr = document.getElementById("tab-mode-ocr");
    const panelViewTitle = document.getElementById("panel-view-title");
    const panelViewSubtitle = document.getElementById("panel-view-subtitle");
    const navZoneElements = document.querySelectorAll(".nav-zone-element");
    const navLegendBar = document.getElementById("nav-legend-bar");
    const navSideSection = document.getElementById("nav-side-section");
    const ocrSideSection = document.getElementById("ocr-side-section");
    const ocrViewfinderGuide = document.getElementById("ocr-viewfinder-guide");
    const ocrActionBar = document.getElementById("ocr-action-bar");
    const focalSettingItem = document.getElementById("focal-setting-item");
    const confSettingItem = document.getElementById("conf-setting-item");

    // Camera Controls
    const btnCameraToggle = document.getElementById("btn-camera-toggle");
    const btnStartHero = document.getElementById("btn-start-cam-hero");
    const btnSwitchCamera = document.getElementById("btn-switch-camera");
    const camBtnIcon = document.getElementById("cam-btn-icon");
    const camBtnText = document.getElementById("cam-btn-text");
    const camPlaceholder = document.getElementById("cam-placeholder");

    // Nav Stats & Banners
    const statFps = document.getElementById("stat-fps");
    const statLatency = document.getElementById("stat-latency");
    const streamStatusDot = document.getElementById("stream-status-dot");
    const streamStatusText = document.getElementById("stream-status-text");
    const activeBanner = document.getElementById("active-alert-banner");
    const alertTitle = document.getElementById("alert-title");
    const alertDesc = document.getElementById("alert-desc");
    const alertBadgeDist = document.getElementById("alert-badge-dist");
    const alertIconBox = document.getElementById("alert-icon-box");

    // Voice & General Controls
    const btnToggleVoice = document.getElementById("btn-toggle-voice");
    const voiceIcon = document.getElementById("voice-icon");
    const srAnnouncer = document.getElementById("sr-announcer");
    const focalSlider = document.getElementById("focal-slider");
    const focalValDisplay = document.getElementById("focal-val-display");
    const confSlider = document.getElementById("conf-slider");
    const confValDisplay = document.getElementById("conf-val-display");

    // Lists & Logs
    const objectsListContainer = document.getElementById("objects-list-container");
    const detectedCount = document.getElementById("detected-count");
    const alertsLogContainer = document.getElementById("alerts-log-container");
    const btnClearLog = document.getElementById("btn-clear-log");

    // OCR Elements
    const btnOcrSnap = document.getElementById("btn-ocr-snap");
    const btnOcrPlayPause = document.getElementById("btn-ocr-play-pause");
    const ocrPlayIcon = document.getElementById("ocr-play-icon");
    const ocrPlayText = document.getElementById("ocr-play-text");
    const btnOcrReplay = document.getElementById("btn-ocr-replay");
    const ocrTextDisplay = document.getElementById("ocr-text-display");
    const ocrStatusStrip = document.getElementById("ocr-status-strip");
    const ocrStatusLabel = document.getElementById("ocr-status-label");
    const ocrStatsBadge = document.getElementById("ocr-stats-badge");
    const btnTextSmaller = document.getElementById("btn-text-smaller");
    const btnTextLarger = document.getElementById("btn-text-larger");
    const btnCopyOcr = document.getElementById("btn-copy-ocr");
    const ocrSpeedSelect = document.getElementById("ocr-speed-select");
    const btnClearOcr = document.getElementById("btn-clear-ocr");

    // Modals
    const classesModal = document.getElementById("classes-modal");
    const btnOpenClassesModal = document.getElementById("btn-open-classes-modal");
    const btnCloseClassesModal = document.getElementById("btn-close-classes-modal");
    const btnSaveClasses = document.getElementById("btn-save-classes");
    const btnSelectAllClasses = document.getElementById("btn-select-all-classes");
    const shortcutsModal = document.getElementById("shortcuts-modal");
    const btnShortcuts = document.getElementById("btn-shortcuts");
    const btnCloseShortcutsModal = document.getElementById("btn-close-shortcuts-modal");
    const btnCloseShortcuts = document.getElementById("btn-close-shortcuts");
    const fileInput = document.getElementById("file-input");

    // AUDIO PLAYERS
    let audioCtx = null;
    const alertAudio = new Audio();
    const ocrAudio = new Audio();

    function playDangerBeep(freq = 880, duration = 0.15) {
        if (!state.voiceEnabled) return;
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        } catch (e) {
            console.error("Audio beep error:", e);
        }
    }

    let isAudioPlaying = false;
    let lastAudioPlayTime = 0;

    function playVietnameseAlert(alert) {
        if (!state.voiceEnabled || !alert || state.mode !== "nav") return;

        if (srAnnouncer && alert.text_vi) {
            srAnnouncer.textContent = alert.text_vi;
        }

        const isDanger = (alert.risk_level === "DANGER");
        const now = Date.now();

        if (isAudioPlaying) {
            if (!isDanger) return;
            alertAudio.pause();
            alertAudio.currentTime = 0;
        } else if (!isDanger && (now - lastAudioPlayTime < 4500)) {
            return;
        }

        if (isDanger) {
            playDangerBeep(980, 0.15);
        }

        if (alert.audio_base64) {
            isAudioPlaying = true;
            alertAudio.src = alert.audio_base64;
            alertAudio.playbackRate = state.speechRate || 1.1;

            alertAudio.onended = () => {
                isAudioPlaying = false;
                lastAudioPlayTime = Date.now();
            };
            alertAudio.onerror = () => {
                isAudioPlaying = false;
                lastAudioPlayTime = Date.now();
            };

            alertAudio.play().catch(err => {
                console.warn("[Second Eye] Autoplay prevented until user interacts with page:", err);
                isAudioPlaying = false;
            });
        }
    }

    async function playVietnameseText(text) {
        if (!state.voiceEnabled || !text) return;
        try {
            const res = await fetch(`/api/tts?text=${encodeURIComponent(text)}`);
            if (res.ok) {
                const data = await res.json();
                if (data.audio_base64) {
                    playVietnameseAlert({
                        text_vi: text,
                        risk_level: "WARNING",
                        audio_base64: data.audio_base64
                    });
                }
            }
        } catch (e) {
            console.warn("TTS fetch error:", e);
        }
    }

    // MODE SWITCHING LOGIC
    function switchMode(newMode) {
        if (state.mode === newMode) return;
        state.mode = newMode;

        if (newMode === "nav") {
            tabModeNav.classList.add("active");
            tabModeNav.setAttribute("aria-selected", "true");
            tabModeOcr.classList.remove("active");
            tabModeOcr.setAttribute("aria-selected", "false");

            panelViewTitle.textContent = "Camera Trực Tiếp & Phân Tích AR";
            panelViewSubtitle.textContent = "Khung phân tích 15 lớp vật thể";
            activeBanner.style.display = "flex";
            navLegendBar.style.display = "flex";
            navSideSection.style.display = "block";
            ocrSideSection.style.display = "none";
            ocrViewfinderGuide.style.display = "none";
            ocrActionBar.style.display = "none";
            focalSettingItem.style.display = "block";
            confSettingItem.style.display = "block";
            navZoneElements.forEach(el => el.style.display = "block");

            // Pause OCR audio if playing
            ocrAudio.pause();
            state.ocrIsPlaying = false;

            playVietnameseText("Đã chuyển sang chế độ tránh vật cản và định vị.");
        } else if (newMode === "ocr") {
            tabModeOcr.classList.add("active");
            tabModeOcr.setAttribute("aria-selected", "true");
            tabModeNav.classList.remove("active");
            tabModeNav.setAttribute("aria-selected", "false");

            panelViewTitle.textContent = "Camera Trực Tiếp & Đọc Văn Bản";
            panelViewSubtitle.textContent = "Trợ lý OCR nhận diện sách, nhãn hàng & tài liệu";
            activeBanner.style.display = "none";
            navLegendBar.style.display = "none";
            navSideSection.style.display = "none";
            ocrSideSection.style.display = "block";
            ocrViewfinderGuide.style.display = "flex";
            ocrActionBar.style.display = "flex";
            focalSettingItem.style.display = "none";
            confSettingItem.style.display = "none";
            navZoneElements.forEach(el => el.style.display = "none");

            // Clear HUD canvas
            hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

            playVietnameseText("Đã chuyển sang chế độ đọc văn bản. Hướng camera vào tài liệu và bấm nút Đọc văn bản hoặc phím T.");
        }
    }

    tabModeNav.addEventListener("click", () => switchMode("nav"));
    tabModeOcr.addEventListener("click", () => switchMode("ocr"));

    // CAMERA INITIALIZATION
    async function startCamera() {
        try {
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            alertAudio.load();
            ocrAudio.load();

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                const isLocal = location.hostname === "localhost" || location.hostname === "127.0.0.1";
                if (!isLocal && location.protocol !== "https:") {
                    alert(
                        "⚠️ TRÌNH DUYỆT YÊU CẦU HTTPS KHI DÙNG CAMERA TRÊN ĐIỆN THOẠI:\n\n" +
                        "Vui lòng truy cập bằng đường dẫn: https://" + location.hostname + ":" + location.port + "\n" +
                        "(Sau đó bấm 'Nâng cao' -> 'Tiếp tục truy cập')"
                    );
                    return;
                } else {
                    alert("Trình duyệt của bạn không hỗ trợ hoặc đang chặn MediaDevices API.");
                    return;
                }
            }

            let stream = null;
            try {
                const constraints = {
                    video: {
                        facingMode: state.facingMode ? { ideal: state.facingMode } : undefined,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                };
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (constraintErr) {
                console.warn("[Second Eye] Retrying with generic video constraints:", constraintErr);
                stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            }

            video.srcObject = stream;
            video.onloadedmetadata = () => { resizeCanvases(); };
            await video.play();

            state.isStreaming = true;
            camPlaceholder.style.display = "none";
            camBtnIcon.textContent = "⏸";
            camBtnText.textContent = "Tạm Dừng";
            btnCameraToggle.classList.replace("btn-primary", "btn-secondary");
            streamStatusDot.classList.add("dot-live");
            streamStatusText.textContent = "Camera đang chạy";

            playVietnameseText(state.mode === "nav" ? "Camera đã khởi động. Đang quét vật cản." : "Camera đã sẵn sàng đọc văn bản.");

            resizeCanvases();
            requestAnimationFrame(processFrameLoop);
        } catch (err) {
            console.error("Camera access error:", err);
            let userMsg = "Không thể mở camera: " + err.message;
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                userMsg = "Trình duyệt chưa được cấp quyền Camera. Hãy bấm vào biểu tượng Ổ khóa trên thanh địa chỉ để Cho phép (Allow).";
            }
            alert(userMsg);
        }
    }

    function stopCamera() {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        state.isStreaming = false;
        camPlaceholder.style.display = "flex";
        camBtnIcon.textContent = "▶";
        camBtnText.textContent = "Bật Camera";
        btnCameraToggle.classList.replace("btn-secondary", "btn-primary");
        streamStatusDot.classList.remove("dot-live");
        streamStatusText.textContent = "Camera đã dừng";
        
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        state.currentObjects = [];
        updateObjectsList([]);
        updateAlertBanner(null);
    }

    function toggleCamera() {
        if (state.isStreaming) stopCamera();
        else startCamera();
    }

    async function switchCamera() {
        state.facingMode = state.facingMode === "environment" ? "user" : "environment";
        if (state.isStreaming) {
            stopCamera();
            await startCamera();
        }
    }

    function resizeCanvases() {
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        hudCanvas.width = vw;
        hudCanvas.height = vh;
    }

    // PROCESSING & INFERENCE LOOP (FOR NAVIGATION MODE)
    const offscreenCanvas = document.createElement("canvas");
    const offCtx = offscreenCanvas.getContext("2d");

    async function processFrameLoop() {
        if (!state.isStreaming) return;

        // In OCR Mode, we don't spam YOLO object detection on every frame to save CPU/Battery
        if (state.mode === "nav") {
            if (!state.isProcessingFrame && video.readyState >= video.HAVE_CURRENT_DATA) {
                state.isProcessingFrame = true;
                await processNavFrame();
                state.isProcessingFrame = false;
            }
        }

        requestAnimationFrame(processFrameLoop);
    }

    async function processNavFrame() {
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 480;
        offCtx.drawImage(video, 0, 0, 640, 480);
        const base64Img = offscreenCanvas.toDataURL("image/jpeg", 0.65);

        try {
            const response = await fetch("/api/detect_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: base64Img,
                    focal_length: state.focalLength,
                    confidence_threshold: state.confidence
                })
            });

            if (response.ok) {
                const data = await response.json();
                statFps.textContent = data.fps || "0.0";
                statLatency.textContent = `${data.inference_ms || 0}ms`;

                state.currentObjects = data.objects || [];
                state.radarObjects = state.currentObjects;

                drawHUD(state.currentObjects);
                updateObjectsList(state.currentObjects);
                renderRadar();

                if (data.active_alert) {
                    updateAlertBanner(data.active_alert);
                    playVietnameseAlert(data.active_alert);
                    addAlertLog(data.active_alert);
                } else if (state.currentObjects.length === 0) {
                    updateAlertBanner(null);
                }
            }
        } catch (err) {
            console.warn("Detection network glitch:", err);
        }
    }

    // OCR TEXT & DOCUMENT READING ENGINE
    async function triggerOCRReading() {
        if (state.isReadingOCR) return;
        if (!state.isStreaming && (!video.srcObject || video.readyState < 2)) {
            alert("Vui lòng Bật Camera trước khi quét văn bản.");
            await startCamera();
            return;
        }

        state.isReadingOCR = true;
        btnOcrSnap.classList.add("loading");
        btnOcrSnap.querySelector(".btn-text").textContent = "ĐANG ĐỌC VĂN BẢN...";
        ocrStatusLabel.textContent = "⏳ Đang phân tích chữ & tổng hợp giọng đọc Tiếng Việt...";

        // Capture high-res frame
        const vw = video.videoWidth || 1280;
        const vh = video.videoHeight || 720;
        offscreenCanvas.width = vw;
        offscreenCanvas.height = vh;
        offCtx.drawImage(video, 0, 0, vw, vh);
        const base64Img = offscreenCanvas.toDataURL("image/jpeg", 0.85);

        try {
            const res = await fetch("/api/ocr/read_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: base64Img,
                    synthesize_audio: true
                })
            });

            if (res.ok) {
                const data = await res.json();
                state.ocrResult = data;
                state.ocrParagraphs = data.audio_paragraphs || [];
                state.ocrCurrentParaIdx = 0;

                // Update Stats
                ocrStatsBadge.textContent = `${data.word_count} từ | ${Math.round(data.avg_confidence * 100)}% độ tin cậy`;

                if (!data.full_text || data.full_text.trim().length === 0) {
                    ocrStatusLabel.textContent = "❌ Không tìm thấy văn bản rõ ràng. Hãy đưa tài liệu lại gần và đủ sáng.";
                    ocrTextDisplay.innerHTML = `
                        <div class="ocr-empty-hint">
                            <span class="hint-icon">🔍</span>
                            <h4>Không nhận diện được văn bản</h4>
                            <p>Hãy giữ thẳng camera, đảm bảo đủ ánh sáng và đưa tài liệu vào khung ngắm.</p>
                        </div>
                    `;
                    playVietnameseText("Không tìm thấy văn bản rõ ràng. Vui lòng thử lại.");
                } else {
                    ocrStatusLabel.textContent = `✅ Đã nhận diện ${data.word_count} từ (${data.paragraphs.length} đoạn). Đang đọc...`;
                    renderOCRTextContent(data.paragraphs);
                    drawOCROverlay(data.lines);
                    startSequentialReading();
                }
            } else {
                ocrStatusLabel.textContent = "Lỗi khi xử lý ảnh.";
            }
        } catch (e) {
            console.error("OCR request error:", e);
            ocrStatusLabel.textContent = "Lỗi kết nối máy chủ OCR.";
        } finally {
            state.isReadingOCR = false;
            btnOcrSnap.classList.remove("loading");
            btnOcrSnap.querySelector(".btn-text").textContent = "ĐỌC VĂN BẢN NÀY (PHÍM T)";
        }
    }

    function renderOCRTextContent(paragraphs) {
        if (!paragraphs || paragraphs.length === 0) return;
        ocrTextDisplay.innerHTML = "";
        paragraphs.forEach((p, idx) => {
            const pEl = document.createElement("p");
            pEl.className = "ocr-para";
            pEl.id = `ocr-para-${idx}`;
            pEl.style.fontSize = `${state.ocrFontSize}rem`;
            pEl.textContent = p;
            pEl.addEventListener("click", () => {
                jumpToParagraph(idx);
            });
            ocrTextDisplay.appendChild(pEl);
        });
    }

    function drawOCROverlay(lines) {
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        if (!lines || lines.length === 0) return;

        const scaleX = hudCanvas.width / (video.videoWidth || hudCanvas.width);
        const scaleY = hudCanvas.height / (video.videoHeight || hudCanvas.height);

        lines.forEach((l, idx) => {
            const [x, y, w, h] = l.rect;
            hudCtx.strokeStyle = "rgba(0, 242, 254, 0.9)";
            hudCtx.lineWidth = 2;
            hudCtx.fillStyle = "rgba(0, 242, 254, 0.15)";
            hudCtx.fillRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
            hudCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

            // Badge
            hudCtx.fillStyle = "#00f2fe";
            hudCtx.beginPath();
            hudCtx.arc(x * scaleX + 8, y * scaleY + 8, 8, 0, Math.PI * 2);
            hudCtx.fill();
            hudCtx.fillStyle = "#000";
            hudCtx.font = "bold 10px JetBrains Mono";
            hudCtx.fillText(String(idx + 1), x * scaleX + 5, y * scaleY + 11);
        });
    }

    function startSequentialReading() {
        if (!state.ocrParagraphs || state.ocrParagraphs.length === 0) return;
        state.ocrCurrentParaIdx = 0;
        playCurrentParagraph();
    }

    function playCurrentParagraph() {
        if (state.ocrCurrentParaIdx >= state.ocrParagraphs.length) {
            state.ocrIsPlaying = false;
            ocrPlayIcon.textContent = "▶";
            ocrPlayText.textContent = "Đọc Lại";
            ocrStatusLabel.textContent = "✅ Đã đọc xong toàn bộ văn bản.";
            document.querySelectorAll(".ocr-para").forEach(el => el.classList.remove("para-active"));
            return;
        }

        const item = state.ocrParagraphs[state.ocrCurrentParaIdx];
        document.querySelectorAll(".ocr-para").forEach(el => el.classList.remove("para-active"));
        const activeParaEl = document.getElementById(`ocr-para-${state.ocrCurrentParaIdx}`);
        if (activeParaEl) {
            activeParaEl.classList.add("para-active");
            activeParaEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        if (item.audio_base64) {
            state.ocrIsPlaying = true;
            ocrPlayIcon.textContent = "⏸️";
            ocrPlayText.textContent = "Tạm Dừng";
            ocrStatusLabel.textContent = `🔊 Đang đọc đoạn ${state.ocrCurrentParaIdx + 1}/${state.ocrParagraphs.length}...`;

            ocrAudio.src = item.audio_base64;
            const speed = parseFloat(ocrSpeedSelect.value) || 1.0;
            ocrAudio.playbackRate = speed;

            ocrAudio.onended = () => {
                state.ocrCurrentParaIdx++;
                playCurrentParagraph();
            };

            ocrAudio.onerror = () => {
                state.ocrCurrentParaIdx++;
                playCurrentParagraph();
            };

            ocrAudio.play().catch(e => {
                console.warn("OCR Audio play prevented:", e);
                state.ocrIsPlaying = false;
            });
        }
    }

    function toggleOCRPlayPause() {
        if (!state.ocrParagraphs || state.ocrParagraphs.length === 0) {
            triggerOCRReading();
            return;
        }

        if (state.ocrIsPlaying) {
            ocrAudio.pause();
            state.ocrIsPlaying = false;
            ocrPlayIcon.textContent = "▶";
            ocrPlayText.textContent = "Tiếp Tục";
            ocrStatusLabel.textContent = "⏸️ Đã tạm dừng đọc.";
        } else {
            if (state.ocrCurrentParaIdx >= state.ocrParagraphs.length) {
                state.ocrCurrentParaIdx = 0;
            }
            playCurrentParagraph();
        }
    }

    function replayOCR() {
        if (!state.ocrParagraphs || state.ocrParagraphs.length === 0) {
            triggerOCRReading();
            return;
        }
        state.ocrCurrentParaIdx = 0;
        playCurrentParagraph();
    }

    function jumpToParagraph(idx) {
        state.ocrCurrentParaIdx = idx;
        playCurrentParagraph();
    }

    // OCR Action Button Listeners
    btnOcrSnap.addEventListener("click", triggerOCRReading);
    btnOcrPlayPause.addEventListener("click", toggleOCRPlayPause);
    btnOcrReplay.addEventListener("click", replayOCR);

    btnTextLarger.addEventListener("click", () => {
        state.ocrFontSize = Math.min(2.2, state.ocrFontSize + 0.15);
        document.querySelectorAll(".ocr-para").forEach(p => p.style.fontSize = `${state.ocrFontSize}rem`);
    });

    btnTextSmaller.addEventListener("click", () => {
        state.ocrFontSize = Math.max(0.9, state.ocrFontSize - 0.15);
        document.querySelectorAll(".ocr-para").forEach(p => p.style.fontSize = `${state.ocrFontSize}rem`);
    });

    btnCopyOcr.addEventListener("click", () => {
        if (state.ocrResult && state.ocrResult.full_text) {
            navigator.clipboard.writeText(state.ocrResult.full_text);
            ocrStatusLabel.textContent = "📋 Đã sao chép văn bản vào clipboard!";
            setTimeout(() => { ocrStatusLabel.textContent = "✅ Văn bản sẵn sàng."; }, 2500);
        }
    });

    btnClearOcr.addEventListener("click", () => {
        ocrAudio.pause();
        state.ocrIsPlaying = false;
        state.ocrResult = null;
        state.ocrParagraphs = [];
        ocrStatsBadge.textContent = "0 từ | 0% độ tin cậy";
        ocrStatusLabel.textContent = "Hướng camera vào tài liệu và bấm [Đọc Văn Bản]";
        ocrTextDisplay.innerHTML = `
            <div class="ocr-empty-hint">
                <span class="hint-icon">📄</span>
                <h4>Chưa có văn bản nào được quét</h4>
                <p>Hướng camera về phía sách, tài liệu in, nhãn thuốc,... và bấm <strong>[Đọc Văn Bản Này]</strong>.</p>
            </div>
        `;
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
    });

    ocrSpeedSelect.addEventListener("change", (e) => {
        const speed = parseFloat(e.target.value) || 1.0;
        ocrAudio.playbackRate = speed;
    });

    // DRAW HUD OVERLAYS (NAVIGATION MODE)
    function drawHUD(objects) {
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        if (state.mode !== "nav" || !objects || objects.length === 0) return;

        const scaleX = hudCanvas.width / 640;
        const scaleY = hudCanvas.height / 480;

        objects.forEach(obj => {
            const [x1, y1, x2, y2] = obj.bbox.map((v, i) => i % 2 === 0 ? v * scaleX : v * scaleY);
            let color = "#10b981";
            if (obj.risk_level === "DANGER") color = "#ef4444";
            else if (obj.risk_level === "WARNING") color = "#f59e0b";

            hudCtx.strokeStyle = color;
            hudCtx.lineWidth = (obj.risk_level === "DANGER") ? 3 : 2;
            hudCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Bounding box corner accents
            const cl = Math.min(15, (x2 - x1) / 4, (y2 - y1) / 4);
            hudCtx.lineWidth = 4;
            hudCtx.beginPath();
            hudCtx.moveTo(x1, y1 + cl); hudCtx.lineTo(x1, y1); hudCtx.lineTo(x1 + cl, y1);
            hudCtx.moveTo(x2 - cl, y1); hudCtx.lineTo(x2, y1); hudCtx.lineTo(x2, y1 + cl);
            hudCtx.moveTo(x1, y2 - cl); hudCtx.lineTo(x1, y2); hudCtx.lineTo(x1 + cl, y2);
            hudCtx.moveTo(x2 - cl, y2); hudCtx.lineTo(x2, y2); hudCtx.lineTo(x2, y2 - cl);
            hudCtx.stroke();

            // Badge text
            const badgeText = `${obj.name_vi.toUpperCase()} | ${obj.distance.toFixed(1)}m`;
            hudCtx.font = "bold 13px 'Outfit', sans-serif";
            const textMetrics = hudCtx.measureText(badgeText);
            const bgW = textMetrics.width + 12;
            const bgH = 20;

            hudCtx.fillStyle = color;
            hudCtx.fillRect(x1, Math.max(0, y1 - bgH), bgW, bgH);
            hudCtx.fillStyle = (obj.risk_level === "WARNING") ? "#000000" : "#ffffff";
            hudCtx.fillText(badgeText, x1 + 6, Math.max(14, y1 - 5));
        });
    }

    // 2D TOP-DOWN RADAR RENDERER
    function renderRadar() {
        radarCtx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
        const cx = radarCanvas.width / 2;
        const cy = radarCanvas.height - 20;
        const maxRange = 4.0; // 4 meters
        const scale = (radarCanvas.height - 40) / maxRange;

        // Radar sweep background rings
        for (let r = 1.0; r <= maxRange; r += 1.0) {
            const rad = r * scale;
            radarCtx.strokeStyle = "rgba(0, 242, 254, 0.15)";
            radarCtx.lineWidth = 1;
            radarCtx.beginPath();
            radarCtx.arc(cx, cy, rad, Math.PI, 2 * Math.PI);
            radarCtx.stroke();

            radarCtx.fillStyle = "rgba(148, 163, 184, 0.5)";
            radarCtx.font = "10px 'JetBrains Mono'";
            radarCtx.fillText(`${r.toFixed(0)}m`, cx + 4, cy - rad + 12);
        }

        // Radar sector divider rays
        [-0.4, 0, 0.4].forEach(angle => {
            radarCtx.strokeStyle = "rgba(0, 242, 254, 0.12)";
            radarCtx.beginPath();
            radarCtx.moveTo(cx, cy);
            radarCtx.lineTo(cx + Math.sin(angle) * maxRange * scale, cy - Math.cos(angle) * maxRange * scale);
            radarCtx.stroke();
        });

        // Dynamic sweep line
        state.radarAngle += 0.04;
        const sweepAngle = Math.sin(state.radarAngle) * 0.7;
        radarCtx.strokeStyle = "rgba(0, 242, 254, 0.4)";
        radarCtx.lineWidth = 2;
        radarCtx.beginPath();
        radarCtx.moveTo(cx, cy);
        radarCtx.lineTo(cx + Math.sin(sweepAngle) * maxRange * scale, cy - Math.cos(sweepAngle) * maxRange * scale);
        radarCtx.stroke();

        // Render detected objects as radar blips
        state.radarObjects.forEach(obj => {
            const [x_lat, z_depth] = obj.coord_3d;
            if (z_depth > maxRange) return;

            const px = cx + (x_lat * scale);
            const py = cy - (z_depth * scale);

            let blipColor = "#10b981";
            if (obj.risk_level === "DANGER") blipColor = "#ef4444";
            else if (obj.risk_level === "WARNING") blipColor = "#f59e0b";

            // Blip glow
            radarCtx.fillStyle = blipColor;
            radarCtx.beginPath();
            radarCtx.arc(px, py, 6, 0, Math.PI * 2);
            radarCtx.fill();

            // Label
            radarCtx.fillStyle = "#ffffff";
            radarCtx.font = "bold 9px 'Outfit'";
            radarCtx.fillText(obj.name_vi, px + 8, py + 3);
        });
    }

    // UI UPDATE HELPERS
    function updateAlertBanner(alert) {
        if (!alert) {
            activeBanner.className = "alert-banner alert-idle";
            alertIconBox.textContent = "🛡️";
            alertTitle.textContent = "Môi trường an toàn";
            alertDesc.textContent = "Chưa phát hiện vật cản trong cự ly nguy hiểm.";
            alertBadgeDist.textContent = "-- m";
            return;
        }

        const isDanger = (alert.risk_level === "DANGER");
        activeBanner.className = `alert-banner ${isDanger ? "alert-danger" : "alert-warning"}`;
        alertIconBox.textContent = isDanger ? "⚠️" : "👀";
        alertTitle.textContent = isDanger ? "CẢNH BÁO NGUY HIỂM!" : "Cảnh Giác Vật Cản";
        alertDesc.textContent = alert.text_vi;
        alertBadgeDist.textContent = `${alert.distance.toFixed(1)} m`;
    }

    function updateObjectsList(objects) {
        detectedCount.textContent = `${objects.length} đối tượng`;
        if (objects.length === 0) {
            objectsListContainer.innerHTML = `<div class="empty-list-msg">Không có vật thể nào trong tầm nhìn</div>`;
            return;
        }

        objectsListContainer.innerHTML = "";
        objects.forEach(obj => {
            const item = document.createElement("div");
            item.className = `object-item obj-risk-${obj.risk_level.toLowerCase()}`;
            item.innerHTML = `
                <div class="obj-main-info">
                    <span class="obj-name">${obj.name_vi.toUpperCase()}</span>
                    <span class="obj-dir">${obj.direction_vi}</span>
                </div>
                <div class="obj-metric-info">
                    <span class="obj-dist font-mono">${obj.distance.toFixed(1)}m</span>
                    <span class="obj-conf font-mono">${Math.round(obj.confidence * 100)}%</span>
                </div>
            `;
            objectsListContainer.appendChild(item);
        });
    }

    function addAlertLog(alert) {
        const timeStr = new Date().toLocaleTimeString('vi-VN');
        const emptyMsg = alertsLogContainer.querySelector(".log-empty-msg");
        if (emptyMsg) emptyMsg.remove();

        const logItem = document.createElement("div");
        logItem.className = `log-item log-${alert.risk_level.toLowerCase()}`;
        logItem.innerHTML = `
            <span class="log-time font-mono">${timeStr}</span>
            <span class="log-text">${alert.text_vi}</span>
        `;
        alertsLogContainer.insertBefore(logItem, alertsLogContainer.firstChild);

        if (alertsLogContainer.children.length > 30) {
            alertsLogContainer.removeChild(alertsLogContainer.lastChild);
        }
    }

    // GENERAL LISTENERS
    btnCameraToggle.addEventListener("click", toggleCamera);
    btnStartHero.addEventListener("click", startCamera);
    btnSwitchCamera.addEventListener("click", switchCamera);

    btnToggleVoice.addEventListener("click", () => {
        state.voiceEnabled = !state.voiceEnabled;
        voiceIcon.textContent = state.voiceEnabled ? "🔊" : "🔇";
        btnToggleVoice.querySelector(".btn-text").textContent = `Giọng nói: ${state.voiceEnabled ? "BẬT" : "TẮT"}`;
        btnToggleVoice.classList.toggle("btn-glass", state.voiceEnabled);
        
        if (state.voiceEnabled) {
            playVietnameseText("Âm thanh cảnh báo đã được bật.");
        } else {
            alertAudio.pause();
            ocrAudio.pause();
            isAudioPlaying = false;
            state.ocrIsPlaying = false;
        }
    });

    focalSlider.addEventListener("input", (e) => {
        state.focalLength = parseFloat(e.target.value);
        focalValDisplay.textContent = `${state.focalLength} px`;
    });

    confSlider.addEventListener("input", (e) => {
        state.confidence = parseFloat(e.target.value);
        confValDisplay.textContent = `${Math.round(state.confidence * 100)}%`;
    });

    btnClearLog.addEventListener("click", () => {
        alertsLogContainer.innerHTML = `<div class="log-empty-msg">Chưa có nhật ký cảnh báo</div>`;
    });

    // MODAL HANDLERS
    btnOpenClassesModal.addEventListener("click", () => { classesModal.style.display = "flex"; });
    btnCloseClassesModal.addEventListener("click", () => { classesModal.style.display = "none"; });
    btnSaveClasses.addEventListener("click", async () => {
        const checkboxes = document.querySelectorAll(".class-checkbox");
        const disabled = [];
        checkboxes.forEach(cb => {
            if (!cb.checked) disabled.push(cb.dataset.key);
        });

        await fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ disabled_classes: disabled })
        });

        classesModal.style.display = "none";
        playVietnameseText("Đã lưu cấu hình danh sách vật thể.");
    });

    btnSelectAllClasses.addEventListener("click", () => {
        document.querySelectorAll(".class-checkbox").forEach(cb => cb.checked = true);
    });

    btnShortcuts.addEventListener("click", () => { shortcutsModal.style.display = "flex"; });
    btnCloseShortcutsModal.addEventListener("click", () => { shortcutsModal.style.display = "none"; });
    btnCloseShortcuts.addEventListener("click", () => { shortcutsModal.style.display = "none"; });

    // UPLOAD IMAGE FILE HANDLER
    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (state.mode === "ocr") {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("synthesize_audio", "true");

            ocrStatusLabel.textContent = "⏳ Đang phân tích chữ trong file ảnh tải lên...";
            try {
                const res = await fetch("/api/ocr/read_file", {
                    method: "POST",
                    body: formData
                });
                if (res.ok) {
                    const data = await res.json();
                    state.ocrResult = data;
                    state.ocrParagraphs = data.audio_paragraphs || [];
                    state.ocrCurrentParaIdx = 0;
                    ocrStatsBadge.textContent = `${data.word_count} từ | ${Math.round(data.avg_confidence * 100)}% độ tin cậy`;

                    renderOCRTextContent(data.paragraphs);
                    if (data.annotated_image) {
                        const img = new Image();
                        img.onload = () => {
                            hudCanvas.width = img.width;
                            hudCanvas.height = img.height;
                            hudCtx.drawImage(img, 0, 0);
                        };
                        img.src = data.annotated_image;
                    }
                    startSequentialReading();
                }
            } catch (err) {
                console.error("Upload OCR error:", err);
            }
        } else {
            const reader = new FileReader();
            reader.onload = async (event) => {
                const img = new Image();
                img.onload = async () => {
                    offscreenCanvas.width = img.width;
                    offscreenCanvas.height = img.height;
                    offCtx.drawImage(img, 0, 0);
                    const base64 = offscreenCanvas.toDataURL("image/jpeg", 0.7);

                    if (state.isStreaming) stopCamera();
                    camPlaceholder.style.display = "none";
                    hudCanvas.width = img.width;
                    hudCanvas.height = img.height;
                    const ctx = hudCanvas.getContext("2d");
                    ctx.drawImage(img, 0, 0);

                    const res = await fetch("/api/detect_frame", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            image: base64,
                            focal_length: state.focalLength,
                            confidence_threshold: state.confidence
                        })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        state.currentObjects = data.objects || [];
                        state.radarObjects = state.currentObjects;
                        drawHUD(state.currentObjects);
                        updateObjectsList(state.currentObjects);
                        renderRadar();
                    }
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // KEYBOARD SHORTCUTS
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;

        switch (e.code) {
            case "Digit1":
                switchMode("nav");
                break;
            case "Digit2":
                switchMode("ocr");
                break;
            case "Space":
                e.preventDefault();
                if (state.mode === "ocr") {
                    triggerOCRReading();
                } else {
                    toggleCamera();
                }
                break;
            case "KeyT":
                if (state.mode !== "ocr") switchMode("ocr");
                triggerOCRReading();
                break;
            case "KeyP":
                if (state.mode === "ocr") toggleOCRPlayPause();
                break;
            case "KeyR":
                if (state.mode === "ocr") {
                    replayOCR();
                } else {
                    if (state.currentObjects.length > 0) {
                        const primary = state.currentObjects[0];
                        playVietnameseText(`Phía trước có ${primary.name_vi}, cách ${primary.distance.toFixed(1)} mét.`);
                    } else {
                        playVietnameseText("Hiện không có vật cản nào được phát hiện.");
                    }
                }
                break;
            case "KeyV":
                btnToggleVoice.click();
                break;
            case "KeyM":
                switchCamera();
                break;
            case "Equal":
                if (state.mode === "nav") {
                    state.focalLength = Math.min(1200, state.focalLength + 25);
                    focalSlider.value = state.focalLength;
                    focalValDisplay.textContent = `${state.focalLength} px`;
                    playVietnameseText(`Tiêu cự ${state.focalLength}`);
                }
                break;
            case "Minus":
                if (state.mode === "nav") {
                    state.focalLength = Math.max(300, state.focalLength - 25);
                    focalSlider.value = state.focalLength;
                    focalValDisplay.textContent = `${state.focalLength} px`;
                    playVietnameseText(`Tiêu cự ${state.focalLength}`);
                }
                break;
            case "Escape":
                if (classesModal.style.display !== "none") classesModal.style.display = "none";
                if (shortcutsModal.style.display !== "none") shortcutsModal.style.display = "none";
                if (state.ocrIsPlaying) ocrAudio.pause();
                break;
        }
    });
});
