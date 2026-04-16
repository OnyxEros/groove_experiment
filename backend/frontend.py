HTML_PAGE = """
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
    animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

h2 {
    text-align: center;
    margin-bottom: 6px;
    font-weight: 600;
}

.subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
}

.counter {
    text-align: center;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 18px;
}

audio {
    width: 100%;
    margin: 10px 0 20px;
    border-radius: 10px;
}

.slider-block {
    margin: 18px 0;
}

label {
    display: block;
    margin-bottom: 6px;
    font-size: 14px;
    color: var(--muted);
}

input[type=range] {
    width: 100%;
    accent-color: var(--accent);
}

button {
    width: 100%;
    padding: 14px;
    margin-top: 18px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #6c8cff, #4c7dff);
    color: white;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.08s ease, opacity 0.2s;
}

button:hover {
    transform: scale(1.02);
}

button:active {
    transform: scale(0.98);
}

button:disabled {
    background: #2a2f3a;
    cursor: not-allowed;
}

.progress {
    height: 6px;
    width: 100%;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 18px;
}

.progress-bar {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.3s ease;
}

</style>
</head>

<body>

<div id="app">
    <h2>🎧 Groove Study</h2>
    <div class="subtitle">Écoute et évalue les extraits</div>

    <div class="progress">
        <div class="progress-bar" id="progress"></div>
    </div>

    <div id="content"></div>
</div>

<script>

let participant_id = null;
let stimuli = [];
let idx = 0;
let start_time = 0;
let is_sending = false;

async function init() {

    participant_id = (await fetch("/new_participant").then(r => r.json())).participant_id;
    stimuli = await fetch("/stimuli?n=20").then(r => r.json());

    render();
}

function render() {

    if (idx >= stimuli.length) {
        document.getElementById("app").innerHTML = `
            <h2>🙏 Merci !</h2>
            <div class="subtitle">Expérience terminée</div>
        `;
        return;
    }

    start_time = Date.now();
    let s = stimuli[idx];

    document.getElementById("progress").style.width =
        ((idx / stimuli.length) * 100) + "%";

    document.getElementById("content").innerHTML = `

        <div class="counter">${idx+1} / ${stimuli.length}</div>

        <audio controls autoplay>
            <source src="${s.audio_url}" type="audio/mp3">
        </audio>

        <div class="slider-block">
            <label>Groove</label>
            <input type="range" id="g" min="1" max="7" value="4">
        </div>

        <div class="slider-block">
            <label>Complexité</label>
            <input type="range" id="c" min="1" max="7" value="4">
        </div>

        <button onclick="send()" id="btn">
            ${is_sending ? "Envoi..." : "Continuer"}
        </button>
    `;
}

async function send() {

    if (is_sending) return;
    is_sending = true;

    let s = stimuli[idx];
    let rt = (Date.now() - start_time) / 1000;

    document.getElementById("btn").disabled = true;

    // ✅ CLEAN PAYLOAD (match Supabase table)
    const payload = {
        participant_id: participant_id,
        stim_id: s.stim_id || s.audio_file,
        groove: Number(document.getElementById("g").value),
        complexity: Number(document.getElementById("c").value),
        rt: isNaN(rt) ? 0 : rt
    };

    await fetch("/response", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    idx++;
    is_sending = false;
    render();
}

init();

</script>

</body>
</html>
"""