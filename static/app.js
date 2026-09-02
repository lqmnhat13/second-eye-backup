/**
 * Second Eye - Frontend Engine (Webcam Stream, Web Speech API, AR HUD & 2D Radar)
 */

document.addEventListener("DOMContentLoaded", () => {
    // STATE
    const state = {
        isStreaming: false,
        facingMode: "environment", // "user" or "environment" (rear camera preferred for navigation)
        focalLength: 650,
        confidence: 0.35,
        voiceEnabled: true,
        speechRate: 1.1,
        dangerThreshold: 1.0,
        warningThreshold: 2.0,
        isProcessingFrame: false,
        disabledClasses: new Set(),
        currentObjects: [],
        lastSpokenPhrase: "",
        lastAlertTime: 0,
        radarAngle: 0,
        radarObjects: []
    };

    // DOM ELEMENTS
    const video = document.getElementById("webcam-video");
    const hudCanvas = document.getElementById("hud-canvas");
    const hudCtx = hudCanvas.getContext("2d");
    const radarCanvas = document.getElementById("radar-canvas");
    const radarCtx = radarCanvas.getContext("2d");

    const btnCameraToggle = document.getElementById("btn-camera-toggle");
    const btnStartHero = document.getElementById("btn-start-cam-hero");
    const btnSwitchCamera = document.getElementById("btn-switch-camera");
    const camBtnIcon = document.getElementById("cam-btn-icon");
    const camBtnText = document.getElementById("cam-btn-text");
    const camPlaceholder = document.getElementById("cam-placeholder");

    const statFps = document.getElementById("stat-fps");
    const statLatency = document.getElementById("stat-latency");
    const streamStatusDot = document.getElementById("stream-status-dot");
    const streamStatusText = document.getElementById("stream-status-text");

    const activeBanner = document.getElementById("active-alert-banner");
    const alertTitle = document.getElementById("alert-title");
    const alertDesc = document.getElementById("alert-desc");
    const alertBadgeDist = document.getElementById("alert-badge-dist");
    const alertIconBox = document.getElementById("alert-icon-box");

    const btnToggleVoice = document.getElementById("btn-toggle-voice");
    const voiceIcon = document.getElementById("voice-icon");
    const srAnnouncer = document.getElementById("sr-announcer");

    const focalSlider = document.getElementById("focal-slider");
    const focalValDisplay = document.getElementById("focal-val-display");
    const confSlider = document.getElementById("conf-slider");
    const confValDisplay = document.getElementById("conf-val-display");
    const speechRateSlider = document.getElementById("speech-rate-slider");
    const speechRateDisplay = document.getElementById("speech-rate-display");

    const objectsListContainer = document.getElementById("objects-list-container");
    const detectedCount = document.getElementById("detected-count");
    const alertsLogContainer = document.getElementById("alerts-log-container");
    const btnClearLog = document.getElementById("btn-clear-log");

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

    // WEB AUDIO SYNTHESIZER (For instant danger beeps)
    let audioCtx = null;
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

    // NATIVE VIETNAMESE AUDIO PLAYER (100% PURE VIETNAMESE MP3)
    let isAudioPlaying = false;
    let lastAudioPlayTime = 0;
    const alertAudio = new Audio();

    function playVietnameseAlert(alert) {
        if (!state.voiceEnabled || !alert) return;

        // Announce for Screen Readers
        if (srAnnouncer && alert.text_vi) {
            srAnnouncer.textContent = alert.text_vi;
        }

        const isDanger = (alert.risk_level === "DANGER");
        const now = Date.now();

        // Strict lock: if audio is currently playing, ignore new normal warnings
        if (isAudioPlaying) {
            if (!isDanger) {
                return; // Drop message, don't spam
            } else {
                // Emergency danger: interrupt current sound
                alertAudio.pause();
                alertAudio.currentTime = 0;
            }
        } else if (!isDanger && (now - lastAudioPlayTime < 4500)) {
            // Enforce minimum 4.5s of silence between voice alerts
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

    // Safe backwards-compatible wrapper
    function speakVietnamese(text) {
        playVietnameseText(text);
    }

    // CAMERA INITIALIZATION
    async function startCamera() {
        try {
            // Unlock audio on initial user interaction (handles mobile autoplay policies)
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            alertAudio.load();

            // Check browser MediaDevices API support (often blocked on non-localhost HTTP)
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                const isLocal = location.hostname === "localhost" || location.hostname === "127.0.0.1";
                if (!isLocal && location.protocol !== "https:") {
                    alert(
                        "⚠️ TRÌNH DUYỆT YÊU CẦU HTTPS KHI DÙNG CAMERA TRÊN ĐIỆN THOẠI:\n\n" +
                        "Trình duyệt di động (iOS Safari / Android Chrome) bắt buộc sử dụng HTTPS để cấp quyền camera.\n\n" +
                        "👉 Cách khắc phục:\n" +
                        "1. Truy cập bằng đường dẫn HTTPS: https://" + location.hostname + ":" + location.port + "\n" +
                        "   (Sau đó bấm 'Nâng cao' / 'Advanced' -> 'Tiếp tục truy cập' / 'Proceed')\n" +
                        "2. Hoặc nếu dùng trên máy tính: mở http://localhost:" + location.port
                    );
                    return;
                } else {
                    alert("Trình duyệt của bạn không hỗ trợ hoặc đang chặn MediaDevices API.");
                    return;
                }
            }

            let stream = null;
            // First attempt: try requested facingMode with ideal constraint (non-strict so laptops won't fail)
            try {
                const constraints = {
                    video: {
                        facingMode: state.facingMode ? { ideal: state.facingMode } : undefined,
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                };
                stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (constraintErr) {
                console.warn("[Second Eye] Retrying with generic video constraints:", constraintErr);
                // Fallback attempt: request any available video camera
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            }

            video.srcObject = stream;
            
            video.onloadedmetadata = () => {
                resizeCanvases();
            };

            await video.play();

            state.isStreaming = true;
            camPlaceholder.style.display = "none";
            camBtnIcon.textContent = "⏸";
            camBtnText.textContent = "Tạm Dừng";
            btnCameraToggle.classList.replace("btn-primary", "btn-secondary");
            streamStatusDot.classList.add("dot-live");
            streamStatusText.textContent = "Camera đang chạy";

            playVietnameseText("Camera đã khởi động. Đang quét vật thể.");

            resizeCanvases();
            requestAnimationFrame(processFrameLoop);
        } catch (err) {
            console.error("Camera access error:", err);
            let userMsg = "Không thể mở camera (" + (err.name || "Lỗi") + ").";
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                userMsg = "Trình duyệt chưa được cấp quyền Camera.\n\n👉 Vui lòng bấm vào biểu tượng Ổ khóa hoặc Cài đặt trang web trên thanh địa chỉ và chọn 'Cho phép' (Allow) Camera.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                userMsg = "Không tìm thấy thiết bị camera nào trên máy.";
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                userMsg = "Camera đang bị một ứng dụng khác (FaceTime, Zoom, Meet, ...) sử dụng. Vui lòng đóng ứng dụng đó và thử lại.";
            } else if (err.name === "OverconstrainedError") {
                userMsg = "Độ phân giải hoặc chế độ camera không được hỗ trợ trên thiết bị này.";
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
        
        // Clear canvases
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
        state.currentObjects = [];
        updateObjectsList([]);
        updateAlertBanner(null);
    }

    function toggleCamera() {
        if (state.isStreaming) {
            stopCamera();
        } else {
            startCamera();
        }
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

    // PROCESSING & INFERENCE LOOP
    const offscreenCanvas = document.createElement("canvas");
    const offCtx = offscreenCanvas.getContext("2d");

    async function processFrameLoop() {
        if (!state.isStreaming) return;

        if (!state.isProcessingFrame && video.readyState >= video.HAVE_CURRENT_DATA) {
            state.isProcessingFrame = true;

            const t0 = performance.now();
            const vw = video.videoWidth || 640;
            const vh = video.videoHeight || 480;
            offscreenCanvas.width = vw;
            offscreenCanvas.height = vh;
            offCtx.drawImage(video, 0, 0, vw, vh);

            const base64Data = offscreenCanvas.toDataURL("image/jpeg", 0.65);

            // Timeout controller (4000ms max per frame request)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);

            try {
                const response = await fetch("/api/detect_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image: base64Data,
                        focal_length: state.focalLength,
                        conf_threshold: state.confidence
                    }),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (response.ok) {
                    const data = await response.json();
                    const t1 = performance.now();
                    const latency = Math.round(t1 - t0);
                    
                    statLatency.textContent = `${latency}ms`;
                    statFps.textContent = data.fps ? data.fps.toFixed(1) : "0.0";

                    // Handle Detections & Rendering
                    state.currentObjects = data.objects || [];
                    state.radarObjects = state.currentObjects;

                    drawHUD(state.currentObjects);
                    updateObjectsList(state.currentObjects);
                    handleAlerts(data.alerts || [], state.currentObjects);
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.warn("Detection request error:", err);
                }
            } finally {
                clearTimeout(timeoutId);
                state.isProcessingFrame = false;
            }
        }

        // Radar animation loop
        renderRadar();

        requestAnimationFrame(processFrameLoop);
    }

    // RENDER HUD AR ON CANVAS
    function drawHUD(objects) {
        hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

        const w = hudCanvas.width;
        const h = hudCanvas.height;

        objects.forEach(obj => {
            const [x1, y1, x2, y2] = obj.bbox;
            const bw = x2 - x1;
            const bh = y2 - y1;

            let color = "#10b981"; // Safe Green
            let glow = "rgba(16, 185, 129, 0.4)";
            if (obj.risk_level === "DANGER") {
                color = "#ef4444"; // Danger Red
                glow = "rgba(239, 68, 68, 0.6)";
            } else if (obj.risk_level === "WARNING") {
                color = "#f59e0b"; // Warning Amber
                glow = "rgba(245, 158, 11, 0.5)";
            }

            // Draw bounding box
            hudCtx.save();
            hudCtx.strokeStyle = color;
            hudCtx.lineWidth = obj.risk_level === "DANGER" ? 3 : 2;
            hudCtx.shadowColor = glow;
            hudCtx.shadowBlur = 8;
            hudCtx.strokeRect(x1, y1, bw, bh);

            // Draw glowing corner brackets
            const cLen = Math.min(20, bw / 4, bh / 4);
            hudCtx.lineWidth = 4;
            // Top-left
            hudCtx.beginPath();
            hudCtx.moveTo(x1, y1 + cLen);
            hudCtx.lineTo(x1, y1);
            hudCtx.lineTo(x1 + cLen, y1);
            hudCtx.stroke();
            // Bottom-right
            hudCtx.beginPath();
            hudCtx.moveTo(x2, y2 - cLen);
            hudCtx.lineTo(x2, y2);
            hudCtx.lineTo(x2 - cLen, y2);
            hudCtx.stroke();

            // Label text
            const label = `${obj.name_vi.toUpperCase()} • ${obj.distance.toFixed(1)}m (${obj.direction_vi})`;
            hudCtx.font = "bold 13px 'Outfit', sans-serif";
            const textWidth = hudCtx.measureText(label).width;
            
            // Label tag background
            hudCtx.fillStyle = color;
            const tagY = Math.max(0, y1 - 22);
            hudCtx.fillRect(x1, tagY, textWidth + 14, 22);

            // Label text
            hudCtx.fillStyle = obj.risk_level === "WARNING" ? "#000" : "#fff";
            hudCtx.shadowBlur = 0;
            hudCtx.fillText(label, x1 + 7, tagY + 16);

            hudCtx.restore();
        });
    }

    // RENDER 2D TOP-DOWN RADAR
    function renderRadar() {
        const rc = radarCtx;
        const w = radarCanvas.width;
        const h = radarCanvas.height;
        const cx = w / 2;
        const cy = h - 35; // User position near bottom
        const maxRangeMeters = 4.0;
        const maxRadiusPx = 110;

        rc.clearRect(0, 0, w, h);

        // Radar background glow
        rc.save();

        // Range rings (1m, 2m, 3m, 4m)
        const rings = [1.0, 2.0, 3.0, 4.0];
        rings.forEach(r => {
            const rad = (r / maxRangeMeters) * maxRadiusPx;
            rc.beginPath();
            rc.arc(cx, cy, rad, Math.PI, 0); // Forward semicircle
            rc.strokeStyle = r === 1.0 ? "rgba(239, 68, 68, 0.35)" : (r === 2.0 ? "rgba(245, 158, 11, 0.25)" : "rgba(0, 242, 254, 0.15)");
            rc.lineWidth = 1;
            rc.stroke();

            // Distance labels
            rc.fillStyle = "rgba(148, 163, 184, 0.6)";
            rc.font = "9px 'JetBrains Mono', monospace";
            rc.fillText(`${r.toFixed(0)}m`, cx + 4, cy - rad + 3);
        });

        // Direction sector lines (Left 35%, Center, Right 65%)
        const angles = [-Math.PI * 0.75, -Math.PI * 0.5, -Math.PI * 0.25];
        angles.forEach(ang => {
            rc.beginPath();
            rc.moveTo(cx, cy);
            rc.lineTo(cx + Math.cos(ang) * maxRadiusPx, cy + Math.sin(ang) * maxRadiusPx);
            rc.strokeStyle = "rgba(255, 255, 255, 0.08)";
            rc.stroke();
        });

        // Animated sweeping beam
        state.radarAngle += 0.04;
        const sweepAngle = -Math.PI + (Math.sin(state.radarAngle) + 1) * 0.5 * Math.PI;
        rc.beginPath();
        rc.moveTo(cx, cy);
        rc.arc(cx, cy, maxRadiusPx, sweepAngle - 0.2, sweepAngle);
        rc.closePath();
        const beamGrad = rc.createRadialGradient(cx, cy, 10, cx, cy, maxRadiusPx);
        beamGrad.addColorStop(0, "rgba(0, 242, 254, 0.3)");
        beamGrad.addColorStop(1, "rgba(0, 242, 254, 0.0)");
        rc.fillStyle = beamGrad;
        rc.fill();

        // Render detected object blips
        state.radarObjects.forEach(obj => {
            const dist = Math.min(maxRangeMeters, obj.distance);
            const distPx = (dist / maxRangeMeters) * maxRadiusPx;
            
            // Map lateral X [-1.5m to 1.5m] to angle
            const latX = obj.coord_3d ? obj.coord_3d[0] : 0;
            const objAngle = -Math.PI / 2 + (latX / 2.0);

            const bx = cx + Math.cos(objAngle) * distPx;
            const by = cy + Math.sin(objAngle) * distPx;

            let bColor = "#10b981";
            if (obj.risk_level === "DANGER") bColor = "#ef4444";
            else if (obj.risk_level === "WARNING") bColor = "#f59e0b";

            // Blip dot
            rc.beginPath();
            rc.arc(bx, by, obj.risk_level === "DANGER" ? 6 : 4, 0, Math.PI * 2);
            rc.fillStyle = bColor;
            rc.shadowColor = bColor;
            rc.shadowBlur = 10;
            rc.fill();

            // Blip label
            rc.fillStyle = "#fff";
            rc.font = "bold 9px 'Outfit', sans-serif";
            rc.fillText(obj.name_vi, bx + 7, by + 3);
        });

        rc.restore();
    }

    // UPDATE DETECTED OBJECTS SIDEBAR
    function updateObjectsList(objects) {
        detectedCount.textContent = `${objects.length} đối tượng`;

        if (objects.length === 0) {
            objectsListContainer.innerHTML = `<div class="empty-list-msg">Không có vật thể nào trong tầm nhìn</div>`;
            return;
        }

        objectsListContainer.innerHTML = objects.map(obj => {
            const riskCls = obj.risk_level.toLowerCase();
            return `
                <div class="object-item ${riskCls}">
                    <div class="object-info">
                        <span class="object-name">${obj.name_vi.toUpperCase()}</span>
                        <span class="object-dir">${obj.direction_vi} • Độ tin cậy: ${Math.round(obj.confidence * 100)}%</span>
                    </div>
                    <div class="object-dist-badge">${obj.distance.toFixed(1)}m</div>
                </div>
            `;
        }).join("");
    }

    // HANDLE ACTIVE BANNER & VOICE ALERTS
    function handleAlerts(serverAlerts, objects) {
        const mostCritical = objects.find(o => o.risk_level === "DANGER") ||
                             objects.find(o => o.risk_level === "WARNING") || null;

        updateAlertBanner(mostCritical);

        if (serverAlerts && serverAlerts.length > 0) {
            serverAlerts.forEach(alert => {
                appendAlertLog(alert);
                // Play authentic native Vietnamese MP3 audio directly
                playVietnameseAlert(alert);
            });
        }
    }

    function updateAlertBanner(obj) {
        if (!obj || obj.risk_level === "SAFE") {
            activeBanner.className = "alert-banner alert-idle";
            alertTitle.textContent = "Môi trường an toàn";
            alertDesc.textContent = "Chưa phát hiện vật cản trong cự ly nguy hiểm.";
            alertBadgeDist.textContent = "-- m";
            alertIconBox.textContent = "🛡️";
            return;
        }

        if (obj.risk_level === "DANGER") {
            activeBanner.className = "alert-banner alert-danger-active";
            alertIconBox.textContent = "⚠️";
            alertTitle.textContent = `NGUY HIỂM: ${obj.name_vi.toUpperCase()} (${obj.direction_vi})`;
            alertDesc.textContent = `Vật cản ở cự ly rất gần! Hãy giảm tốc độ hoặc đổi hướng.`;
            alertBadgeDist.textContent = `${obj.distance.toFixed(1)}m`;
        } else if (obj.risk_level === "WARNING") {
            activeBanner.className = "alert-banner alert-warning-active";
            alertIconBox.textContent = "👁️";
            alertTitle.textContent = `CHÚ Ý: Có ${obj.name_vi} ${obj.direction_vi.toLowerCase()}`;
            alertDesc.textContent = `Vật cản trong cự ly cần chú ý (${obj.distance.toFixed(1)}m).`;
            alertBadgeDist.textContent = `${obj.distance.toFixed(1)}m`;
        }
    }

    function appendAlertLog(alert) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const riskCls = alert.risk_level.toLowerCase();

        const entry = document.createElement("div");
        entry.className = `log-entry ${riskCls}`;
        entry.innerHTML = `
            <span class="log-time">[${timeStr}]</span>
            <span class="log-text">${alert.text_vi}</span>
        `;

        if (alertsLogContainer.querySelector(".log-empty-msg")) {
            alertsLogContainer.innerHTML = "";
        }

        alertsLogContainer.prepend(entry);
        if (alertsLogContainer.children.length > 30) {
            alertsLogContainer.removeChild(alertsLogContainer.lastChild);
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

    // CONTROLS & SETTINGS LISTENERS
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
            alertAudio.currentTime = 0;
            isAudioPlaying = false;
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

    speechRateSlider.addEventListener("input", (e) => {
        state.speechRate = parseFloat(e.target.value);
        speechRateDisplay.textContent = `${state.speechRate.toFixed(1)}x`;
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

    // UPLOAD IMAGE/VIDEO FILE HANDLER
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            const img = new Image();
            img.onload = async () => {
                offscreenCanvas.width = img.width;
                offscreenCanvas.height = img.height;
                offCtx.drawImage(img, 0, 0);
                const base64 = offscreenCanvas.toDataURL("image/jpeg", 0.7);

                // Stop live stream if active
                if (state.isStreaming) stopCamera();

                camPlaceholder.style.display = "none";
                hudCanvas.width = img.width;
                hudCanvas.height = img.height;

                // Draw static image on canvas
                const ctx = hudCanvas.getContext("2d");
                ctx.drawImage(img, 0, 0);

                const res = await fetch("/api/detect_frame", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image: base64,
                        focal_length: state.focalLength,
                        conf_threshold: state.confidence
                    })
                });

                if (res.ok) {
                    const data = await res.json();
                    state.currentObjects = data.objects || [];
                    state.radarObjects = state.currentObjects;
                    drawHUD(state.currentObjects);
                    updateObjectsList(state.currentObjects);
                    handleAlerts(data.alerts || [], state.currentObjects);
                    renderRadar();
                }
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    // KEYBOARD ACCESSIBILITY SHORTCUTS
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;

        switch (e.code) {
            case "Space":
                e.preventDefault();
                toggleCamera();
                break;
            case "KeyV":
                btnToggleVoice.click();
                break;
            case "KeyR":
                if (state.currentObjects.length > 0) {
                    const primary = state.currentObjects[0];
                    playVietnameseText(`Phía trước có ${primary.name_vi}, cách ${primary.distance.toFixed(1)} mét.`);
                } else {
                    playVietnameseText("Hiện không có vật cản nào được phát hiện.");
                }
                break;
            case "Equal": // Key '+'
                state.focalLength = Math.min(1200, state.focalLength + 25);
                focalSlider.value = state.focalLength;
                focalValDisplay.textContent = `${state.focalLength} px`;
                playVietnameseText(`Tiêu cự ${state.focalLength}`);
                break;
            case "Minus": // Key '-'
                state.focalLength = Math.max(300, state.focalLength - 25);
                focalSlider.value = state.focalLength;
                focalValDisplay.textContent = `${state.focalLength} px`;
                playVietnameseText(`Tiêu cự ${state.focalLength}`);
                break;
            case "KeyM":
                switchCamera();
                break;
            case "Escape":
                if (classesModal.style.display !== "none") classesModal.style.display = "none";
                if (shortcutsModal.style.display !== "none") shortcutsModal.style.display = "none";
                if (state.isStreaming) stopCamera();
                break;
        }
    });
});
