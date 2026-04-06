/* ═══════════════════════════════════════════════════════════════
   SecureAI-Guard — SOC Dashboard Logic
   Connects to FastAPI backend at localhost:7860
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:7860";

// ──── State ────
const state = {
    connected: false,
    episodeActive: false,
    running: false,
    currentObs: null,
    currentState: null,
    eventHistory: [],
    rewardHistory: [],
    stepCount: 0,
};

// ──── DOM refs ────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    statusPill:    $("#status-pill"),
    statusText:    $("#status-text"),
    headerEpisode: $("#header-episode"),
    headerSteps:   $("#header-steps"),
    headerScore:   $("#header-score"),
    taskSelect:    $("#task-select"),
    seedInput:     $("#seed-input"),
    btnReset:      $("#btn-reset"),
    btnAutoStep:   $("#btn-auto-step"),
    btnRunAll:     $("#btn-run-all"),
    btnHumanStep:  $("#btn-human-step"),
    btnNewEpisode: $("#btn-new-episode"),
    confSlider:    $("#confidence-slider"),
    confValue:     $("#confidence-value"),
    reasoning:     $("#reasoning-input"),
    eventTbody:    $("#event-tbody"),
    currentEvent:  $("#current-event-card"),
    gradeBanner:   $("#grade-banner"),
    taskInfo:      $("#task-info-section"),
    taskInfoContent: $("#task-info-content"),
    // Gauges
    trustRing:   $("#trust-ring"),
    trustValue:  $("#trust-value"),
    trustSub:    $("#trust-sub"),
    fatigueRing: $("#fatigue-ring"),
    fatigueValue:$("#fatigue-value"),
    fatigueSub:  $("#fatigue-sub"),
    riskRing:    $("#risk-ring"),
    riskValue:   $("#risk-value"),
    riskSub:     $("#risk-sub"),
    // Stats
    statThreats:     $("#stat-threats"),
    statFP:          $("#stat-fp"),
    statTotalReward: $("#stat-total-reward"),
    statDifficulty:  $("#stat-difficulty"),
    // Reward bars
    barSecurity:  $("#bar-security"),
    barFriction:  $("#bar-friction"),
    barDelay:     $("#bar-delay"),
    barReasoning: $("#bar-reasoning"),
    valSecurity:  $("#val-security"),
    valFriction:  $("#val-friction"),
    valDelay:     $("#val-delay"),
    valReasoning: $("#val-reasoning"),
    // Grade
    gradeIcon:    $("#grade-icon"),
    gradeTitle:   $("#grade-title"),
    gradeScore:   $("#grade-score"),
    gradeLetter:  $("#grade-letter"),
    gradePassed:  $("#grade-passed"),
    // Event preview
    eventChannelBadge: $("#event-channel-badge"),
    eventThreatBadge:  $("#event-threat-badge"),
    eventSender:       $("#event-sender"),
    eventContent:      $("#event-content"),
    eventRiskScore:    $("#event-risk-score"),
    // Chart
    rewardCanvas:  $("#reward-chart"),
};

// ──── Gauge constants ────
const CIRCUMFERENCE = 2 * Math.PI * 52; // ~326.73

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    checkHealth();
    initChart();
    updateGauges(100, 0, 0);
});

function initEventListeners() {
    dom.btnReset.addEventListener("click", handleReset);
    dom.btnAutoStep.addEventListener("click", handleAutoStep);
    dom.btnRunAll.addEventListener("click", handleRunAll);
    dom.btnHumanStep.addEventListener("click", handleHumanStep);
    dom.btnNewEpisode.addEventListener("click", handleReset);

    dom.confSlider.addEventListener("input", () => {
        dom.confValue.textContent = parseFloat(dom.confSlider.value).toFixed(2);
    });

    // Radio card selection
    $$(".radio-card").forEach((card) => {
        card.addEventListener("click", () => {
            $$(".radio-card").forEach((c) => c.classList.remove("active"));
            card.classList.add("active");
            card.querySelector("input").checked = true;
        });
    });

    dom.taskSelect.addEventListener("change", loadTaskInfo);
}

// ══════════════════════════════════════════════════════════════
// API CALLS
// ══════════════════════════════════════════════════════════════

async function apiCall(endpoint, method = "GET", body = null) {
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);

    const resp = await fetch(`${API_BASE}${endpoint}`, opts);
    if (!resp.ok) throw new Error(`API ${endpoint}: ${resp.status} ${resp.statusText}`);
    return resp.json();
}

async function checkHealth() {
    try {
        const data = await apiCall("/health");
        setStatus("connected", `Online — v${data.version}`);
        state.connected = true;
        loadTaskInfo();
    } catch {
        setStatus("error", "Backend Offline");
        state.connected = false;
        setTimeout(checkHealth, 5000);
    }
}

// ══════════════════════════════════════════════════════════════
// HANDLERS
// ══════════════════════════════════════════════════════════════

async function handleReset() {
    if (!state.connected) return toast("Backend is offline", "error");

    const taskId = dom.taskSelect.value;
    const seed = parseInt(dom.seedInput.value) || 42;

    dom.btnReset.disabled = true;
    setStatus("running", "Resetting environment…");

    try {
        const data = await apiCall("/reset", "POST", { task_id: taskId, seed });
        state.episodeActive = true;
        state.currentObs = data.observation;
        state.currentState = data.state;
        state.eventHistory = [];
        state.rewardHistory = [];
        state.stepCount = 0;

        dom.btnAutoStep.disabled = false;
        dom.btnRunAll.disabled = false;
        dom.btnHumanStep.disabled = false;
        dom.gradeBanner.style.display = "none";

        updateGauges(
            data.state.user_trust,
            data.state.system_fatigue,
            data.observation.hf_risk_score * 100
        );
        updateStats(data.state);
        updateEventPreview(data.observation);
        clearEventTable();
        clearChart();

        dom.headerEpisode.textContent = data.state.episode_id.substring(0, 8);
        dom.headerSteps.textContent = "0";
        dom.headerScore.textContent = "0.0000";

        setStatus("connected", "Episode active");
        toast(`Episode started — ${taskId}`, "success");
    } catch (err) {
        toast(`Reset failed: ${err.message}`, "error");
        setStatus("error", "Reset failed");
    } finally {
        dom.btnReset.disabled = false;
    }
}

async function handleAutoStep() {
    if (!state.episodeActive || state.running) return;
    state.running = true;
    dom.btnAutoStep.disabled = true;
    dom.btnRunAll.disabled = true;

    const action = buildAIAction();
    await executeStep(action);

    state.running = false;
    if (state.episodeActive) {
        dom.btnAutoStep.disabled = false;
        dom.btnRunAll.disabled = false;
    }
}

async function handleRunAll() {
    if (!state.episodeActive || state.running) return;
    state.running = true;
    dom.btnAutoStep.disabled = true;
    dom.btnRunAll.disabled = true;
    dom.btnHumanStep.disabled = true;
    setStatus("running", "Running full episode…");

    while (state.episodeActive && state.running) {
        const action = buildAIAction();
        await executeStep(action);
        await sleep(200); // slight delay for visual effect
    }

    state.running = false;
    if (state.episodeActive) {
        dom.btnAutoStep.disabled = false;
        dom.btnRunAll.disabled = false;
        dom.btnHumanStep.disabled = false;
    }
}

async function handleHumanStep() {
    if (!state.episodeActive || state.running) return;

    const decision = document.querySelector('input[name="decision"]:checked')?.value || "investigate";
    const confidence = parseFloat(dom.confSlider.value);
    const reasoning = dom.reasoning.value.trim() || "Manual override — no reasoning provided.";

    state.running = true;
    dom.btnHumanStep.disabled = true;

    await executeStep({ decision, confidence, reasoning });

    state.running = false;
    if (state.episodeActive) {
        dom.btnHumanStep.disabled = false;
    }
}

async function executeStep(action) {
    try {
        const data = await apiCall("/step", "POST", { action });
        state.currentObs = data.observation;
        state.currentState = data.state;
        state.stepCount++;

        const reward = data.reward.value;
        state.rewardHistory.push(reward);

        // Update UI
        updateGauges(
            data.state.user_trust,
            data.state.system_fatigue,
            data.observation.hf_risk_score * 100
        );
        updateStats(data.state);
        updateRewardBars(data.reward.components);
        addEventRow(data, action);
        updateEventPreview(data.observation);
        addChartPoint(reward);

        dom.headerSteps.textContent = data.state.step_count;
        dom.headerScore.textContent = data.state.total_reward.toFixed(4);

        // Episode done?
        if (data.done) {
            state.episodeActive = false;
            dom.btnAutoStep.disabled = true;
            dom.btnRunAll.disabled = true;
            dom.btnHumanStep.disabled = true;
            state.running = false;

            if (data.grade) {
                showGrade(data.grade);
            }
            setStatus("connected", "Episode complete");
            toast("Episode finished!", "info");
        }
    } catch (err) {
        toast(`Step failed: ${err.message}`, "error");
        state.running = false;
    }
}

// ══════════════════════════════════════════════════════════════
// AI ACTION BUILDER
// ══════════════════════════════════════════════════════════════

function buildAIAction() {
    const obs = state.currentObs;
    if (!obs) return { decision: "investigate", confidence: 0.5, reasoning: "No observation available." };

    const risk = obs.hf_risk_score || 0;
    const trust = obs.user_trust || 100;
    const content = (obs.content || "").toLowerCase();
    const sender = (obs.sender || "").toLowerCase();

    // ── Keyword analysis (PRIMARY signal) ──
    const highRiskKW = ["click here", "verify your account", "suspended", "claim now", "download", "install",
        "wire transfer", "credentials", "emergency", "send money", "customs fee", "stranded",
        "you've won", "password expired", "password has expired", "immediate action", "urgent", "pre-approved",
        "reset here", "action required", "confirm your identity"];
    const medRiskKW = ["http://", "https://", "link", "verify", "update", "confirm",
        "invoice", "document", "offer", "free", "loan", "prize", "exe", ".zip", ".msi"];

    // Safe-content patterns — messages matching these are very likely legitimate
    const safePatterns = ["meeting confirmed", "has shipped", "expected delivery",
        "get back to you", "dinner tonight", "appointment", "monthly statement",
        "conference room", "thanks for", "don't forget", "reminder:", "order #",
        "shipped", "end of day", "see you", "good morning", "good evening",
        "thank you", "happy birthday"];

    // Safe senders
    const safeSenders = ["mom@", "dad@", "boss@", "friend@", "calendar@",
        "no-reply@bank.com", "noreply@amazon.com", "@mycompany.com", "@gmail.com"];

    const highHits = highRiskKW.filter(kw => content.includes(kw)).length;
    const medHits  = medRiskKW.filter(kw => content.includes(kw)).length;
    const safeHits = safePatterns.filter(p => content.includes(p)).length;
    const isSafeSender = safeSenders.some(s => sender.includes(s));

    let decision, confidence, reasoning;

    // ── Decision logic: keywords FIRST, risk score as tiebreaker ──

    // 1. Strong threat indicators → BLOCK
    if (highHits >= 2 || (highHits >= 1 && medHits >= 2)) {
        decision = "block";
        confidence = 0.92;
        reasoning = `High risk — ${highHits} high-risk + ${medHits} medium-risk keyword indicators detected. Blocking to protect user from likely phishing/social-engineering.`;
    }
    // 2. Moderate threat indicators → WARN
    else if (highHits >= 1 || medHits >= 3) {
        decision = "warn";
        confidence = 0.75;
        reasoning = `Moderate risk — ${highHits} high + ${medHits} medium keyword indicators. Issuing warning while preserving user trust.`;
    }
    // 3. Clearly safe content with no risk keywords → ALLOW
    else if (safeHits >= 1 && highHits === 0 && medHits === 0) {
        decision = "allow";
        confidence = 0.90;
        reasoning = `Message appears legitimate — matched ${safeHits} safe pattern(s), no risk keywords. Safe to allow.`;
    }
    // 4. Known safe sender + no risk keywords → ALLOW
    else if (isSafeSender && highHits === 0 && medHits <= 1) {
        decision = "allow";
        confidence = 0.85;
        reasoning = `Trusted sender (${obs.sender}), no significant risk keywords. Allowing message.`;
    }
    // 5. Some medium keywords → INVESTIGATE
    else if (medHits >= 1) {
        decision = "investigate";
        confidence = 0.60;
        reasoning = `Low-moderate risk — ${medHits} medium indicator(s), risk_score=${risk.toFixed(2)}. Flagging for investigation.`;
    }
    // 6. High HF risk score BUT no keywords — use investigate (not block!)
    //    The HF model is sentiment-based and can give false positives.
    else if (risk > 0.7 && !isSafeSender && safeHits === 0) {
        decision = "investigate";
        confidence = 0.55;
        reasoning = `ML model flagged elevated risk (score=${risk.toFixed(2)}) but no keyword indicators found. Investigating as precaution.`;
    }
    // 7. Default: no indicators at all → ALLOW
    else {
        decision = "allow";
        confidence = 0.88;
        reasoning = `Low risk — no threat indicators detected, risk_score=${risk.toFixed(2)}. Message appears safe.`;
    }

    // Trust pressure: if trust is critically low, be more cautious
    if (trust < 25 && decision === "allow" && !isSafeSender && safeHits === 0) {
        decision = "warn";
        confidence = 0.65;
        reasoning += " [Trust critically low — upgrading to warn as precaution.]";
    }

    return { decision, confidence, reasoning };
}

// ══════════════════════════════════════════════════════════════
// UI UPDATES
// ══════════════════════════════════════════════════════════════

function setStatus(type, text) {
    dom.statusPill.className = `status-pill ${type}`;
    dom.statusText.textContent = text;
}

function updateGauges(trust, fatigue, risk) {
    setGauge(dom.trustRing, trust, dom.trustValue, dom.trustSub, "trust");
    setGauge(dom.fatigueRing, fatigue, dom.fatigueValue, dom.fatigueSub, "fatigue");
    setGauge(dom.riskRing, risk, dom.riskValue, dom.riskSub, "risk");
}

function setGauge(ring, value, valueEl, subEl, type) {
    const pct = Math.max(0, Math.min(100, value));
    const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
    ring.style.strokeDashoffset = offset;

    valueEl.textContent = Math.round(pct);

    // Color and label by type
    if (type === "trust") {
        const color = pct > 70 ? "var(--green)" : pct > 35 ? "var(--orange)" : "var(--red)";
        const label = pct > 70 ? "Excellent" : pct > 35 ? "Moderate" : "Critical";
        ring.style.stroke = color;
        subEl.textContent = label;
        subEl.style.color = color;
    } else if (type === "fatigue") {
        const color = pct < 30 ? "var(--green)" : pct < 65 ? "var(--orange)" : "var(--red)";
        const label = pct < 30 ? "Low" : pct < 65 ? "Moderate" : "High";
        ring.style.stroke = color;
        subEl.textContent = label;
        subEl.style.color = color;
    } else if (type === "risk") {
        const color = pct < 30 ? "var(--green)" : pct < 60 ? "var(--orange)" : "var(--red)";
        const label = pct < 30 ? "Safe" : pct < 60 ? "Elevated" : "Danger";
        ring.style.stroke = color;
        subEl.textContent = label;
        subEl.style.color = color;
    }
}

function updateStats(s) {
    dom.statThreats.textContent = s.blocked_threats;
    dom.statFP.textContent = s.false_positives;
    dom.statTotalReward.textContent = s.total_reward.toFixed(4);
    dom.statDifficulty.textContent = s.adversarial_drift_active ? "L3 DRIFT" : "—";
}

function updateRewardBars(comp) {
    if (!comp) return;

    const maxVal = 1.0;
    const setBar = (barEl, valEl, value) => {
        const absVal = Math.abs(value);
        const pct = Math.min((absVal / maxVal) * 100, 100);
        barEl.style.width = `${pct}%`;
        valEl.textContent = value.toFixed(3);
    };

    setBar(dom.barSecurity, dom.valSecurity, comp.security || 0);
    setBar(dom.barFriction, dom.valFriction, comp.user_friction || 0);
    setBar(dom.barDelay, dom.valDelay, comp.delay || 0);
    setBar(dom.barReasoning, dom.valReasoning, comp.reasoning_quality || 0);
}

function updateEventPreview(obs) {
    if (!obs) {
        dom.currentEvent.style.display = "none";
        return;
    }
    dom.currentEvent.style.display = "block";
    dom.eventSender.textContent = obs.sender;
    dom.eventContent.textContent = obs.content;
    dom.eventRiskScore.textContent = (obs.hf_risk_score * 100).toFixed(1) + "%";

    const channel = obs.channel;
    dom.eventChannelBadge.textContent = channel.toUpperCase();
    dom.eventChannelBadge.className = `badge channel-${channel}`;

    const threatType = obs.metadata?.event_type || "unknown";
    dom.eventThreatBadge.textContent = threatType.replace("_", " ").toUpperCase();
    dom.eventThreatBadge.className = `badge threat-${threatType}`;
}

function addEventRow(data, action) {
    // Remove empty row if present
    const emptyRow = dom.eventTbody.querySelector(".empty-row");
    if (emptyRow) emptyRow.remove();

    const tr = document.createElement("tr");
    tr.classList.add("flash");

    const obs = data.observation;
    const reward = data.reward.value;
    const threatType = data.info?.threat_type || "—";

    const decisionClass = `decision-${action.decision}`;
    const rewardClass = reward >= 0 ? "reward-positive" : "reward-negative";

    tr.innerHTML = `
        <td>${data.state.step_count}</td>
        <td>${new Date().toLocaleTimeString()}</td>
        <td><span class="badge channel-${obs.channel}" style="font-size:0.68rem;">${obs.channel.toUpperCase()}</span></td>
        <td title="${obs.sender}">${obs.sender.substring(0, 22)}${obs.sender.length > 22 ? '…' : ''}</td>
        <td title="${obs.content}">${obs.content.substring(0, 45)}${obs.content.length > 45 ? '…' : ''}</td>
        <td><span class="badge threat-${threatType}" style="font-size:0.65rem;">${threatType}</span></td>
        <td><span class="${decisionClass}">${action.decision.toUpperCase()}</span></td>
        <td><span class="${rewardClass}">${reward >= 0 ? '+' : ''}${reward.toFixed(4)}</span></td>
    `;

    // Insert at top
    dom.eventTbody.prepend(tr);

    // Keep max 30 rows
    while (dom.eventTbody.children.length > 30) {
        dom.eventTbody.removeChild(dom.eventTbody.lastChild);
    }

    state.eventHistory.push({ obs, action, reward, threatType });
}

function clearEventTable() {
    dom.eventTbody.innerHTML = `
        <tr class="empty-row">
            <td colspan="8">Waiting for first event…</td>
        </tr>
    `;
}

function showGrade(grade) {
    dom.gradeBanner.style.display = "flex";

    const icons = { S: "🏆", A: "⭐", B: "✅", C: "📊", D: "⚠️", F: "❌" };
    dom.gradeIcon.textContent = icons[grade.grade] || "📋";
    dom.gradeTitle.textContent = grade.passed ? "Mission Accomplished!" : "Mission Failed";
    dom.gradeScore.textContent = grade.score.toFixed(4);
    dom.gradeLetter.textContent = grade.grade;
    dom.gradePassed.textContent = grade.passed ? "Yes ✓" : "No ✕";
    dom.gradePassed.style.color = grade.passed ? "var(--green)" : "var(--red)";
}

async function loadTaskInfo() {
    try {
        const taskId = dom.taskSelect.value;
        const data = await apiCall(`/tasks/${taskId}`);
        dom.taskInfo.style.display = "block";
        dom.taskInfoContent.innerHTML = `
            <p>${data.description}</p>
            <div class="task-meta">
                <span><strong>Difficulty:</strong> ${data.difficulty}</span>
                <span><strong>Max Steps:</strong> ${data.max_steps}</span>
                <span><strong>Threshold:</strong> ${(data.success_threshold * 100).toFixed(0)}%</span>
            </div>
        `;
    } catch {
        dom.taskInfo.style.display = "none";
    }
}

// ══════════════════════════════════════════════════════════════
// SIMPLE CANVAS CHART
// ══════════════════════════════════════════════════════════════

let chartCtx = null;

function initChart() {
    const canvas = dom.rewardCanvas;
    chartCtx = canvas.getContext("2d");
    canvas.width = canvas.parentElement?.clientWidth || 300;
    canvas.height = 160;
    drawChart();
}

function clearChart() {
    state.rewardHistory = [];
    drawChart();
}

function addChartPoint(reward) {
    drawChart();
}

function drawChart() {
    const ctx = chartCtx;
    if (!ctx) return;

    const canvas = ctx.canvas;
    const w = canvas.width;
    const h = canvas.height;
    const data = state.rewardHistory;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "rgba(0,0,0,0.2)";
    ctx.fillRect(0, 0, w, h);

    if (data.length < 2) {
        ctx.fillStyle = "rgba(255,255,255,0.15)";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Reward trend will appear here…", w / 2, h / 2);
        return;
    }

    // Calculate bounds
    const minR = Math.min(...data, -0.5);
    const maxR = Math.max(...data, 0.5);
    const range = maxR - minR || 1;
    const padding = 20;

    // Zero line
    const zeroY = padding + ((maxR - 0) / range) * (h - 2 * padding);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Data line
    const stepX = (w - 2 * padding) / Math.max(data.length - 1, 1);

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, "rgba(0, 212, 255, 0.2)");
    gradient.addColorStop(1, "rgba(0, 212, 255, 0.0)");

    ctx.beginPath();
    ctx.moveTo(padding, padding + ((maxR - data[0]) / range) * (h - 2 * padding));

    for (let i = 1; i < data.length; i++) {
        const x = padding + i * stepX;
        const y = padding + ((maxR - data[i]) / range) * (h - 2 * padding);
        ctx.lineTo(x, y);
    }

    // Fill under the curve
    const lastX = padding + (data.length - 1) * stepX;
    ctx.lineTo(lastX, h);
    ctx.lineTo(padding, h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line on top
    ctx.beginPath();
    ctx.moveTo(padding, padding + ((maxR - data[0]) / range) * (h - 2 * padding));
    for (let i = 1; i < data.length; i++) {
        const x = padding + i * stepX;
        const y = padding + ((maxR - data[i]) / range) * (h - 2 * padding);
        ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Latest point dot
    if (data.length > 0) {
        const lastY = padding + ((maxR - data[data.length - 1]) / range) * (h - 2 * padding);
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#00d4ff";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(lastX, lastY, 7, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0,212,255,0.4)";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// ══════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════

function toast(message, type = "info") {
    const container = $("#toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = "0";
        el.style.transform = "translateY(10px)";
        el.style.transition = "all 0.3s ease";
        setTimeout(() => el.remove(), 300);
    }, 3500);
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// Handle window resize for chart
window.addEventListener("resize", () => {
    if (dom.rewardCanvas && dom.rewardCanvas.parentElement) {
        dom.rewardCanvas.width = dom.rewardCanvas.parentElement.clientWidth - 40;
        drawChart();
    }
});
