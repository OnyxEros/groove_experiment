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

/* ───────────────── INIT ───────────────── */
async function init() {
    try {
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

/* ───────────────── SCREEN CONTROL (FIXED) ───────────────── */
function showScreen(id) {
    document.querySelectorAll('.screen').forEach(el => {
        el.classList.remove('active');
    });

    const target = document.getElementById(id);
    if (target) target.classList.add('active');

    // progress bar uniquement sur task
    const progressWrap = document.getElementById('progress-wrap');
    if (progressWrap) {
        progressWrap.style.display = (id === 'screen-task') ? 'block' : 'none';
    }
}

/* ───────────────── INTRO FLOW ───────────────── */
function goCalibration() {
    showScreen('screen-calib');
    syncSlider(document.getElementById('cg'), 'cg-val');
    syncSlider(document.getElementById('cc'), 'cc-val');
}

function goIntro() {
    showScreen('screen-intro');
}

function startExperiment() {
    showScreen('screen-task');
    render();
}

/* ───────────────── SLIDERS ───────────────── */
function syncSlider(input, pillId) {
    if (!input) return;

    const min = Number(input.min);
    const max = Number(input.max);
    const val = Number(input.value);

    const pct = ((val - min) / (max - min)) * 100;
    input.style.setProperty('--fill', pct + '%');

    const pill = document.getElementById(pillId);
    if (pill) pill.textContent = val;
}

/* ───────────────── RENDER TRIAL ───────────────── */
function render() {
    if (idx >= stimuli.length) {
        showThanks();
        return;
    }

    listenedEnough = false;
    canRespond = false;

    const s = stimuli[idx];
    const pct = Math.round((idx / stimuli.length) * 100);

    const p1 = document.getElementById('progress');
    const l1 = document.getElementById('counter-left');
    const r1 = document.getElementById('counter-right');

    if (p1) p1.style.width = pct + '%';
    if (l1) l1.textContent = `Extrait ${idx + 1} / ${stimuli.length}`;
    if (r1) r1.textContent = pct + '%';

    if (idx + 1 < stimuli.length) preloadOne(stimuli[idx + 1]);

    const content = document.getElementById('content');
    if (content) content.innerHTML = buildTrialHTML(s, idx);

    mountPlayer(s.audio_url);
}

/* ───────────────── PLAYER ───────────────── */
function mountPlayer() {
    const audio = document.getElementById('audio');
    if (!audio) return;

    currentAudio = audio;

    audio.addEventListener('canplaythrough', onCanPlay, { once: true });
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('error', onAudioError);

    audio.play().catch(() => {
        const hint = document.getElementById('autoplay-hint');
        if (hint) hint.classList.add('visible');
    });
}

function onCanPlay() {
    start_time = Date.now();
}

/* ───────────────── AUDIO CHECK ───────────────── */
setInterval(() => {
    const a = currentAudio;
    if (!a || a.paused || !a.duration) return;

    const ok = (a.currentTime / a.duration >= 0.2) || (a.currentTime >= 3);

    if (ok && !listenedEnough) {
        listenedEnough = true;
        canRespond = true;

        const btn = document.getElementById('btn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Continuer →';
        }
    }
}, 400);

/* ───────────────── SEND ───────────────── */
async function send() {
    if (!canRespond || is_sending) return;

    is_sending = true;

    const s = stimuli[idx];
    const rt = (Date.now() - start_time) / 1000;

    const payload = {
        participant_id,
        stim_id: s.stim_id || String(idx),
        groove: Number(document.getElementById('g')?.value),
        complexity: Number(document.getElementById('c')?.value),
        rt,
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
        console.error(e);
        is_sending = false;
        return;
    }

    idx++;

    document.getElementById('content').innerHTML = `<div class="fixation">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 500);
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

function showError(msg) {
    document.getElementById('app').innerHTML =
        `<div class="card" style="text-align:center">${msg}</div>`;
}

function showThanks() {
    document.getElementById('screen-task').innerHTML =
        `<div class="thanks">Merci 🙏</div>`;
}

/* ───────────────── BOOT ───────────────── */
init();