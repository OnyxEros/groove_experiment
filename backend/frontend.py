<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

:root {
    --bg: #0b0c10;
    --card: #15171c;
    --accent: #6c8cff;
    --text: #ffffff;
    --muted: rgba(255,255,255,0.6);
}

body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
    background: radial-gradient(circle at top, #1a1d25, var(--bg));
    color: var(--text);
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

#app {
    width: 92%;
    max-width: 520px;
    background: var(--card);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

h2 { text-align:center; }

.subtitle {
    text-align:center;
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
    line-height: 1.4;
}

button {
    width:100%;
    padding:14px;
    border:none;
    border-radius:12px;
    background: linear-gradient(135deg,#6c8cff,#4c7dff);
    color:white;
    font-size:16px;
    font-weight:600;
    cursor:pointer;
    margin-top: 18px;
}

button:disabled { background:#2a2f3a; }

.progress {
    height:6px;
    background: rgba(255,255,255,0.08);
    border-radius:10px;
    overflow:hidden;
    margin-bottom:18px;
}

.progress-bar {
    height:100%;
    background: var(--accent);
    width:0%;
    transition: width 0.3s ease;
}

.slider-block { margin:18px 0; }

label { font-size:14px; color:var(--muted); }

.scale-labels {
    display:flex;
    justify-content:space-between;
    font-size:11px;
    color:rgba(255,255,255,0.5);
    margin-bottom:4px;
}

input[type=range] {
    width:100%;
    accent-color: var(--accent);
}

.card {
    padding:14px;
    background:rgba(255,255,255,0.04);
    border-radius:12px;
    font-size:14px;
    line-height:1.5;
}

.center { text-align:center; }

</style>
</head>

<body>

<div id="app">

<!-- INTRO -->
<div id="intro">

    <h2>🎧 Groove Study</h2>

    <div class="subtitle">
        Expérience de perception musicale
    </div>

    <div class="card">
        Vous allez écouter des extraits musicaux.
        Après chaque extrait, vous évaluerez :
        <br><br>

        <b>Groove</b> : envie de bouger (taper du pied, hocher la tête)<br><br>

        <b>Complexité</b> : impression de structure rythmique simple ou riche
        <br><br>

        Répondez spontanément — il n’y a pas de bonne ou mauvaise réponse.
    </div>

    <button onclick="startCalibration()">Commencer</button>
</div>

<!-- CALIBRATION -->
<div id="calibration" style="display:none;">
    <h2>Calibration</h2>

    <div class="subtitle">
        Utilisez les sliders selon votre ressenti, sans répondre trop vite.
    </div>

    <button onclick="startExperiment()">Démarrer l'expérience</button>
</div>

<!-- TASK -->
<div id="task" style="display:none;">

    <div class="progress">
        <div class="progress-bar" id="progress"></div>
    </div>

    <div class="subtitle" id="counter"></div>

    <div id="content"></div>
</div>

</div>

<script>

let participant_id = null;
let stimuli = [];
let idx = 0;
let start_time = 0;
let canRespond = false;
let is_sending = false;

/* ---------------- INIT ---------------- */

async function init() {

    participant_id = (await fetch("/new_participant").then(r => r.json())).participant_id;

    stimuli = await fetch("/stimuli?n=20").then(r => r.json());

    shuffle(stimuli);
    preloadAudio(stimuli);
}

/* ---------------- UX FLOW ---------------- */

function startCalibration() {
    document.getElementById("intro").style.display = "none";
    document.getElementById("calibration").style.display = "block";
}

function startExperiment() {
    document.getElementById("calibration").style.display = "none";
    document.getElementById("task").style.display = "block";
    render();
}

/* ---------------- EXPERIMENT ---------------- */

function render() {

    if (idx >= stimuli.length) {
        document.getElementById("app").innerHTML = `
            <h2>🙏 Merci</h2>
            <div class="subtitle">Expérience terminée</div>
        `;
        return;
    }

    let s = stimuli[idx];

    let progress = Math.round((idx / stimuli.length) * 100);
    document.getElementById("progress").style.width = progress + "%";

    document.getElementById("counter").innerText =
        `Extrait ${idx+1} / ${stimuli.length} (${progress}%)`;

    canRespond = false;

    document.getElementById("content").innerHTML = `
        <audio id="audio" controls autoplay>
            <source src="${s.audio_url}" type="audio/mp3">
        </audio>

        <div class="card center" style="margin-top:10px;">
            Écoutez puis évaluez votre ressenti
        </div>

        <div class="slider-block">
            <label>Groove</label>
            <div class="scale-labels"><span>Faible</span><span>Fort</span></div>
            <input type="range" id="g" min="1" max="7" value="4">
        </div>

        <div class="slider-block">
            <label>Complexité</label>
            <div class="scale-labels"><span>Simple</span><span>Complexe</span></div>
            <input type="range" id="c" min="1" max="7" value="4">
        </div>

        <button onclick="send()" id="btn">Continuer</button>
    `;

    let audio = document.getElementById("audio");

    audio.oncanplaythrough = () => {
        start_time = Date.now();
        canRespond = true;
    };
}

/* ---------------- SEND RESPONSE ---------------- */

async function send() {

    if (!canRespond || is_sending) return;
    is_sending = true;

    let s = stimuli[idx];
    let rt = (Date.now() - start_time) / 1000;

    document.getElementById("btn").disabled = true;

    const payload = {
        participant_id,
        stim_id: s.stim_id || s.audio_file,
        groove: Number(document.getElementById("g").value),
        complexity: Number(document.getElementById("c").value),
        rt,
        order: idx,
        timestamp: Date.now(),
        is_catch: s.is_catch || false
    };

    await fetch("/response", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    idx++;

    // inter-stimulus fixation
    document.getElementById("content").innerHTML =
        `<div class="center" style="opacity:0.5; font-size:20px;">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 600);
}

/* ---------------- UTILITIES ---------------- */

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function preloadAudio(stimuli) {
    stimuli.forEach(s => {
        const a = new Audio();
        a.src = s.audio_url;
    });
}

/* ---------------- START ---------------- */

init();

</script>

</body>
</html>