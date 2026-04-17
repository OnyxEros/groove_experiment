let participant_id = null;
let stimuli = [];
let idx = 0;
let start_time = 0;
let canRespond = false;
let is_sending = false;

/* ---------------- INIT ---------------- */

async function init() {
    try {
        const p = await fetch("/new_participant").then(r => r.json());
        participant_id = p.participant_id;

        stimuli = await fetch("/stimuli?n=20").then(r => r.json());
        shuffle(stimuli);

    } catch (e) {
        alert("Erreur de chargement. Recharge la page.");
        console.error(e);
    }
}

/* ---------------- START ---------------- */

function start() {
    document.getElementById("intro").style.display = "none";
    document.getElementById("task").style.display = "block";
    render();
}

/* ---------------- RENDER ---------------- */

function render() {

    if (idx >= stimuli.length) {
        document.getElementById("app").innerHTML =
            "<h2>Merci 🙏</h2><div class='subtitle'>Expérience terminée</div>";
        return;
    }

    let s = stimuli[idx];

    let progress = Math.round((idx / stimuli.length) * 100);
    document.getElementById("progress").style.width = progress + "%";

    document.getElementById("counter").innerText =
        `Extrait ${idx+1} / ${stimuli.length}`;

    canRespond = false;

    document.getElementById("content").innerHTML = `
        <audio id="audio" controls autoplay>
            <source src="${s.audio_url}">
        </audio>

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

        <button onclick="send()" id="btn" disabled>Écoute en cours...</button>
    `;

    let audio = document.getElementById("audio");

    audio.oncanplaythrough = () => {
        start_time = Date.now();

        // petit délai pour éviter réponses instantanées
        setTimeout(() => {
            canRespond = true;
            document.getElementById("btn").disabled = false;
            document.getElementById("btn").innerText = "Continuer";
        }, 800);
    };
}

/* ---------------- SEND ---------------- */

async function send() {

    if (!canRespond || is_sending) return;

    is_sending = true;

    let s = stimuli[idx];
    let rt = (Date.now() - start_time) / 1000;

    let btn = document.getElementById("btn");
    btn.disabled = true;
    btn.innerText = "Envoi...";

    const payload = {
        participant_id: participant_id,
        stim_id: s.stim_id || s.audio_file,
        groove: Number(document.getElementById("g").value),
        complexity: Number(document.getElementById("c").value),
        rt: rt,
        order: idx,
        timestamp: Date.now()
    };

    try {
        await fetch("/response", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

    } catch (e) {
        console.error("Erreur envoi:", e);
        alert("Problème réseau, réessaye.");
        is_sending = false;
        btn.disabled = false;
        btn.innerText = "Continuer";
        return;
    }

    idx++;

    // fixation croix (important cognitivement)
    document.getElementById("content").innerHTML =
        "<div style='text-align:center;opacity:0.5;font-size:24px;'>+</div>";

    setTimeout(() => {
        is_sending = false;
        render();
    }, 500);
}

/* ---------------- UTIL ---------------- */

function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
}

/* ---------------- START ---------------- */

init();