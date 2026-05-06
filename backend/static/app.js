'use strict';

/* ───────────────── STATE ───────────────── */
let participant_id = null;
let stimuli = [];
let idx = 0;
let start_time = 0;
let canRespond = false;
let is_sending = false;
let currentAudio = null;
let listenedEnough = false;
let listenInterval = null;

/* ───────────────── INIT ───────────────── */
document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        showScreen('screen-consent');
        bindConsent();

        const p = await fetch('/new_participant').then(r => r.json());
        participant_id = p.participant_id;

        stimuli = await fetch('/stimuli?n=30').then(r => r.json());
        shuffle(stimuli);

        if (stimuli.length > 0) preloadOne(stimuli[0]);

    } catch (e) {
        console.error('Init error:', e);
        showError('Erreur de chargement — recharge la page.');
    }
}

/* ───────────────── CONSENT ───────────────── */
function bindConsent() {
    const checkbox = document.getElementById('consent-check');
    const btn = document.getElementById('consent-btn');

    if (!checkbox || !btn) return;

    checkbox.addEventListener('change', () => {
        btn.disabled = !checkbox.checked;
    });
}

/* ───────────────── SCREEN CONTROL ───────────────── */
function showScreen(id) {
    document.querySelectorAll('.screen').forEach(el => el.classList.remove('active'));

    const target = document.getElementById(id);
    if (target) target.classList.add('active');

    const progressWrap = document.getElementById('progress-wrap');
    if (progressWrap) {
        progressWrap.style.display = (id === 'screen-task') ? 'block' : 'none';
    }
}

/* ───────────────── FLOW ───────────────── */
function goIntro() { showScreen('screen-intro'); }

function goCalibration() {
    showScreen('screen-calib');
    syncSlider(document.getElementById('cg'), 'cg-val');
    syncSlider(document.getElementById('cc'), 'cc-val');
}

function startExperiment() {
    showScreen('screen-task');
    render();
}

/* ───────────────── RENDER ───────────────── */
function render() {
    if (idx >= stimuli.length) return showThanks();

    listenedEnough = false;
    canRespond = false;

    const s = stimuli[idx];

    const content = document.getElementById('content');
    if (content) content.innerHTML = buildTrialHTML(s);

    mountPlayer();
}

/* ───────────────── AUDIO ───────────────── */
function mountPlayer() {
    const audio = document.getElementById('audio');
    const btn = document.getElementById('btn');

    if (!audio) return;

    currentAudio = audio;

    audio.onplay = () => {
        start_time = Date.now();
    };

    audio.onerror = () => {
        console.error('Audio error');
        btn.textContent = 'Erreur audio → continuer';
        btn.disabled = false;
        canRespond = true;
    };

    // interval sécurisé
    listenInterval = setInterval(() => {
        if (!audio.duration) return;

        const ok = (audio.currentTime / audio.duration >= 0.2) || (audio.currentTime >= 3);

        if (ok && !listenedEnough) {
            listenedEnough = true;
            canRespond = true;

            btn.disabled = false;
            btn.textContent = 'Continuer →';
        }
    }, 300);

    // fallback bouton play
    btn.onclick = () => {
        if (audio.paused) {
            audio.play();
        } else if (canRespond) {
            send();
        }
    };
}

/* ───────────────── SEND ───────────────── */
async function send() {
    if (!canRespond || is_sending) return;

    is_sending = true;

    const s = stimuli[idx];

    const payload = {
        participant_id,
        stim_id: s.stim_id || String(idx),
        rt: (Date.now() - start_time) / 1000,
        trial_index: idx,
        timestamp_client: Date.now()
    };

    try {
        await fetch('/response', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
    } catch (e) {
        console.error('Send error:', e);
    }

    clearInterval(listenInterval);

    idx++;

    const content = document.getElementById('content');
    if (content) content.innerHTML = `<div class="fixation">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 400);
}

/* ───────────────── HELPERS ───────────────── */
function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
}

function preloadOne(s) {
    if (!s?.audio_url) return;
    const a = new Audio();
    a.src = s.audio_url;
}

function buildTrialHTML(s) {
    return `
        <audio id="audio" src="${s.audio_url}" preload="auto"></audio>
        <button id="btn" class="btn">▶ Lancer l'écoute</button>
    `;
}

function showError(msg) {
    document.getElementById('app').innerHTML = `<div class="card">${msg}</div>`;
}

function showThanks() {
    const t = document.getElementById('screen-task');
    if (t) t.innerHTML = `<div class="thanks">Merci 🙏</div>`;
}
