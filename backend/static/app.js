'use strict';

/* ── State ──────────────────────────────────────────────── */
let participant_id = null;
let stimuli        = [];
let idx            = 0;
let start_time     = 0;
let canRespond     = false;
let is_sending     = false;
let currentAudio   = null;

/* ── Init ───────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', init);

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

/* ── Consent ────────────────────────────────────────────── */
function onConsentChange() {
    const checkbox = document.getElementById('consent-check');
    const btn      = document.getElementById('consent-btn');
    if (checkbox && btn) {
        btn.disabled = !checkbox.checked;
    }
}

/* ── Example loader ─────────────────────────────────────── */
function loadExample() {
    const container = document.getElementById('example-container');
    const audio     = document.getElementById('ex-audio');
    if (!container || !audio) return;

    // Fetch a single stimulus to use as the example, or use a static URL if available
    fetch('/stimuli?n=1')
        .then(r => r.json())
        .then(data => {
            if (!data || data.length === 0) {
                container.innerHTML = '<div class="example-loading">Aucun exemple disponible.</div>';
                return;
            }
            const s = data[0];
            audio.src = s.audio_url;
            audio.load();

            container.innerHTML = `
                <div class="player" id="ex-player" style="width:100%">
                    <button class="play-btn" id="ex-play-btn" onclick="toggleExample()">▶</button>
                    <div class="player-info">
                        <div class="player-meta">
                            <span class="player-title">Exemple</span>
                            <span class="player-time" id="ex-time">--:--</span>
                        </div>
                        <div class="player-bar-track">
                            <div class="player-bar-fill" id="ex-fill"></div>
                        </div>
                    </div>
                </div>`;

            audio.addEventListener('timeupdate', () => {
                if (!audio.duration) return;
                const pct  = (audio.currentTime / audio.duration) * 100;
                const fill = document.getElementById('ex-fill');
                const t    = document.getElementById('ex-time');
                if (fill) fill.style.width = pct + '%';
                if (t)    t.textContent = formatTime(audio.currentTime);
            });

            audio.addEventListener('ended', () => {
                const btn = document.getElementById('ex-play-btn');
                if (btn) btn.textContent = '▶';
                const player = document.getElementById('ex-player');
                if (player) player.classList.remove('playing');
            });
        })
        .catch(() => {
            if (container) container.innerHTML = '<div class="example-loading">Exemple non disponible.</div>';
        });
}

function toggleExample() {
    const audio = document.getElementById('ex-audio');
    const btn   = document.getElementById('ex-play-btn');
    const player = document.getElementById('ex-player');
    if (!audio) return;

    if (audio.paused) {
        audio.play().then(() => {
            if (btn)    btn.textContent = '⏸';
            if (player) player.classList.add('playing');
        }).catch(() => {});
    } else {
        audio.pause();
        if (btn)    btn.textContent = '▶';
        if (player) player.classList.remove('playing');
    }
}

/* ── Navigation ─────────────────────────────────────────── */
function goCalibration() {
    showScreen('screen-calib');
    syncSlider(document.getElementById('cg'), 'cg-val');
    syncSlider(document.getElementById('cc'), 'cc-val');
}

function goIntro() {
    showScreen('screen-intro');
}

function startExperiment() {
    const progressWrap = document.getElementById('progress-wrap');
    if (progressWrap) progressWrap.style.display = 'block';
    showScreen('screen-task');
    render();
}

/* ── Screen switcher ────────────────────────────────────── */
function showScreen(id) {
    ['screen-consent', 'screen-intro', 'screen-calib', 'screen-task'].forEach(s => {
        const el = document.getElementById(s);
        if (el) el.style.display = 'none';
    });

    const target = document.getElementById(id);
    if (target) target.style.display = 'block';
}

/* ── Slider sync ───────────────────────────────────────── */
function syncSlider(input, pillId) {
    if (!input) return;

    const min  = Number(input.min);
    const max  = Number(input.max);
    const val  = Number(input.value);
    const pct  = ((val - min) / (max - min)) * 100;

    input.style.setProperty('--fill', pct + '%');

    const pill = document.getElementById(pillId);
    if (pill) pill.textContent = val;
}

/* ── Render trial ───────────────────────────────────────── */
function render() {
    if (idx >= stimuli.length) {
        showThanks();
        return;
    }

    const s   = stimuli[idx];
    const pct = Math.round((idx / stimuli.length) * 100);

    const progress = document.getElementById('progress');
    const left     = document.getElementById('counter-left');
    const right    = document.getElementById('counter-right');

    if (progress) progress.style.width = pct + '%';
    if (left)     left.textContent  = `Extrait ${idx + 1} / ${stimuli.length}`;
    if (right)    right.textContent = pct + '%';

    canRespond = false;
    start_time = 0;

    if (idx + 1 < stimuli.length) preloadOne(stimuli[idx + 1]);

    const content = document.getElementById('content');
    if (content) content.innerHTML = buildTrialHTML(s, idx);

    mountPlayer();

    // Init slider fill after render
    ['g', 'c'].forEach(id => {
        const el = document.getElementById(id);
        if (el) syncSlider(el, id === 'g' ? 'gv' : 'cv');
    });
}

/* ── Trial HTML ─────────────────────────────────────────── */
function buildTrialHTML(s, i) {
    return `
    <div>
        <div class="player waiting" id="player">
            <button class="play-btn" id="play-btn" onclick="togglePlay()">▶</button>

            <div class="player-info">
                <div class="player-meta">
                    <span class="player-title">Extrait ${i + 1}</span>
                    <span class="player-time" id="player-time">--:--</span>
                </div>
                <div class="player-bar-track" onclick="seekTo(event)">
                    <div class="player-bar-fill" id="player-fill"></div>
                </div>
            </div>
        </div>

        <div class="autoplay-hint" id="autoplay-hint">
            ▶ Appuie pour écouter
        </div>

        <audio id="audio" preload="auto">
            <source src="${escHtml(s.audio_url)}" type="audio/mpeg">
        </audio>

        <div style="margin-top: 28px;">
            <div class="slider-block">
                <div class="slider-header">
                    <span class="slider-label">Groove</span>
                    <span class="slider-pill" id="gv">4</span>
                </div>
                <div class="slider-anchors">
                    <span>Aucune envie de bouger</span>
                    <span>Forte envie de bouger</span>
                </div>
                <input type="range" id="g" min="1" max="7" value="4"
                       oninput="syncSlider(this,'gv')">
            </div>

            <div class="slider-block">
                <div class="slider-header">
                    <span class="slider-label">Complexité</span>
                    <span class="slider-pill" id="cv">4</span>
                </div>
                <div class="slider-anchors">
                    <span>Très simple</span>
                    <span>Très complexe</span>
                </div>
                <input type="range" id="c" min="1" max="7" value="4"
                       oninput="syncSlider(this,'cv')">
            </div>
        </div>

        <button class="btn" onclick="send()" id="btn" disabled>
            Écoute en cours…
        </button>
    </div>`;
}

/* ── Player ─────────────────────────────────────────────── */
function mountPlayer() {
    const audio = document.getElementById('audio');
    currentAudio = audio;

    if (!audio) return;

    audio.addEventListener('play', () => {
        if (!start_time) start_time = Date.now();
        const player = document.getElementById('player');
        const btn    = document.getElementById('play-btn');
        if (player) { player.classList.remove('waiting'); player.classList.add('playing'); }
        if (btn)    btn.textContent = '⏸';
    });

    audio.addEventListener('pause', () => {
        const player = document.getElementById('player');
        const btn    = document.getElementById('play-btn');
        if (player) player.classList.remove('playing');
        if (btn)    btn.textContent = '▶';
    });

    audio.addEventListener('timeupdate', () => {
        if (!audio.duration) return;

        const pct  = (audio.currentTime / audio.duration) * 100;
        const fill = document.getElementById('player-fill');
        if (fill) fill.style.width = pct + '%';

        const t = document.getElementById('player-time');
        if (t) t.textContent = formatTime(audio.currentTime);
    });

    audio.addEventListener('ended', () => {
        const btn    = document.getElementById('play-btn');
        const player = document.getElementById('player');
        if (btn)    btn.textContent = '▶';
        if (player) player.classList.remove('playing');
    });

    audio.addEventListener('canplay', () => {
        // Unlock the continue button once audio is ready and autoplay fires
    });

    audio.addEventListener('error', () => {
        canRespond = true;
        enableBtn();
    });

    audio.play()
        .then(() => {
            // Autoplay succeeded — unlock after a short listen
            scheduleUnlock();
        })
        .catch(() => {
            const hint = document.getElementById('autoplay-hint');
            if (hint) hint.classList.add('visible');
        });
}

function scheduleUnlock() {
    setTimeout(() => {
        canRespond = true;
        enableBtn();
    }, 800);
}

function enableBtn() {
    const b = document.getElementById('btn');
    if (b) {
        b.disabled = false;
        b.textContent = 'Continuer →';
    }
}

function togglePlay() {
    const audio = currentAudio;
    if (!audio) return;

    if (audio.paused) {
        audio.play().then(() => {
            if (!canRespond) scheduleUnlock();
        }).catch(() => {});
    } else {
        audio.pause();
    }
}

function seekTo(e) {
    const audio = currentAudio;
    if (!audio || !audio.duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const pct  = (e.clientX - rect.left) / rect.width;

    audio.currentTime = pct * audio.duration;
}

/* ── Send ───────────────────────────────────────────────── */
async function send() {
    if (!canRespond || is_sending) return;

    is_sending = true;

    const s  = stimuli[idx];
    const rt = (Date.now() - start_time) / 1000;

    const payload = {
        participant_id:   participant_id,
        stim_id:          s.stim_id || s.audio_file || String(idx),
        groove:           Number(document.getElementById('g').value),
        complexity:       Number(document.getElementById('c').value),
        rt:               parseFloat(rt.toFixed(3)),
        trial_index:      idx,
        timestamp_client: Date.now()
    };

    try {
        await fetch('/response', {
            method:  'POST',
            headers: {'Content-Type': 'application/json'},
            body:    JSON.stringify(payload)
        });
    } catch (e) {
        console.error('Send error:', e);
        is_sending = false;
        return;
    }

    // Stop audio before transition
    if (currentAudio) { currentAudio.pause(); currentAudio.src = ''; }

    idx++;

    const content = document.getElementById('content');
    if (content) content.innerHTML = `<div class="fixation">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 500);
}

/* ── Helpers ───────────────────────────────────────────── */
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

function formatTime(sec) {
    if (!isFinite(sec)) return '--:--';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function showError(msg) {
    const app = document.getElementById('app');
    if (app) app.innerHTML = `<div class="card">${msg}</div>`;
}

function showThanks() {
    const task = document.getElementById('screen-task');
    if (task) task.innerHTML = `
        <div class="thanks">
            <div class="thanks-icon">🙏</div>
            <h2>Merci pour ta participation !</h2>
            <p>Tes réponses ont bien été enregistrées.<br>
               Tu peux fermer cette page.</p>
            <div class="thanks-detail">participant · ${participant_id}</div>
        </div>`;
}