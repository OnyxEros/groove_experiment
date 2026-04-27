/* ============================================================
   app.js — Groove Study
   ============================================================ */

'use strict';

/* ── State ──────────────────────────────────────────────── */
let participant_id = null;
let stimuli        = [];
let idx            = 0;
let start_time     = 0;
let canRespond     = false;
let is_sending     = false;
let currentAudio   = null;      // HTMLAudioElement actif

/* ── Init ───────────────────────────────────────────────── */
async function init() {
    try {
        const p  = await fetch('/new_participant').then(r => r.json());
        participant_id = p.participant_id;

        stimuli = await fetch('/stimuli?n=20').then(r => r.json());
        shuffle(stimuli);

        // Préchargement discret du premier audio
        if (stimuli.length > 0) preloadOne(stimuli[0]);

    } catch (e) {
        console.error('Init error:', e);
        showError('Erreur de chargement. Recharge la page.');
    }
}

/* ── Start ──────────────────────────────────────────────── */
function startExperiment() {
    document.getElementById('intro').style.display = 'none';

    const task = document.getElementById('task');
    task.style.display = 'block';
    task.classList.add('screen');

    render();
}

/* ── Render ─────────────────────────────────────────────── */
function render() {
    if (idx >= stimuli.length) {
        showThanks();
        return;
    }

    const s = stimuli[idx];
    const pct = Math.round((idx / stimuli.length) * 100);

    // Progress
    document.getElementById('progress').style.width    = pct + '%';
    document.getElementById('counter-left').textContent = `Extrait ${idx + 1} / ${stimuli.length}`;
    document.getElementById('counter-right').textContent = pct + '%';

    canRespond = false;

    // Préchargement du suivant
    if (idx + 1 < stimuli.length) preloadOne(stimuli[idx + 1]);

    document.getElementById('content').innerHTML = buildTrialHTML(s, idx);

    // Bind player
    mountPlayer(s.audio_url);
}

/* ── Build trial HTML ───────────────────────────────────── */
function buildTrialHTML(s, i) {
    return `
        <div class="screen">
            <!-- Player -->
            <div class="player" id="player">
                <button class="play-btn" id="play-btn" onclick="togglePlay()" aria-label="Lecture">
                    ▶
                </button>
                <div class="player-info">
                    <div class="player-title">Extrait ${i + 1}</div>
                    <div class="player-bar-track">
                        <div class="player-bar-fill" id="player-fill"></div>
                    </div>
                    <div class="waveform" id="waveform">${buildWaveform()}</div>
                </div>
            </div>

            <audio id="audio" preload="auto">
                <source src="${escHtml(s.audio_url)}" type="audio/mpeg">
            </audio>

            <!-- Sliders -->
            <div class="slider-block">
                <div class="slider-header">
                    <span class="slider-label">Groove</span>
                    <span class="slider-value" id="gv">4</span>
                </div>
                <div class="scale-labels">
                    <span>Faible</span>
                    <span>Fort</span>
                </div>
                <input type="range" id="g" min="1" max="7" value="4"
                       oninput="document.getElementById('gv').textContent=this.value">
            </div>

            <div class="slider-block">
                <div class="slider-header">
                    <span class="slider-label">Complexité</span>
                    <span class="slider-value" id="cv">4</span>
                </div>
                <div class="scale-labels">
                    <span>Simple</span>
                    <span>Complexe</span>
                </div>
                <input type="range" id="c" min="1" max="7" value="4"
                       oninput="document.getElementById('cv').textContent=this.value">
            </div>

            <button class="btn" onclick="send()" id="btn" disabled>
                Écoute en cours…
            </button>
        </div>
    `;
}

/* ── Waveform decoration ────────────────────────────────── */
function buildWaveform() {
    const heights = [4, 7, 10, 14, 10, 16, 10, 7, 12, 16, 10, 7, 12, 8, 5];
    return heights.map((h, i) =>
        `<span style="height:${h}px;--d:${(0.5 + i * 0.07).toFixed(2)}s"></span>`
    ).join('');
}

/* ── Audio player ───────────────────────────────────────── */
function mountPlayer(url) {
    const audio = document.getElementById('audio');
    currentAudio = audio;

    audio.addEventListener('canplaythrough', onCanPlay, { once: true });
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('error', onAudioError);

    // Autoplay tentative (navigateurs modernes bloquent sans interaction)
    audio.play().catch(() => {
        /* silencieux — l'utilisateur cliquera sur play */
    });
}

function onCanPlay() {
    start_time = Date.now();
    setTimeout(() => {
        canRespond = true;
        const btn = document.getElementById('btn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Continuer →';
        }
    }, 800);

    const player = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    if (player) player.classList.add('playing');
    if (playBtn) { playBtn.textContent = '⏸'; playBtn.classList.add('playing'); }
}

function onTimeUpdate() {
    const audio = currentAudio;
    if (!audio || !audio.duration) return;
    const pct = (audio.currentTime / audio.duration) * 100;
    const fill = document.getElementById('player-fill');
    if (fill) fill.style.width = pct + '%';
}

function onEnded() {
    const player  = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    if (player)  player.classList.remove('playing');
    if (playBtn) { playBtn.textContent = '▶'; playBtn.classList.remove('playing'); }
}

function onAudioError() {
    const btn = document.getElementById('btn');
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Continuer →';
        canRespond = true;
    }
    console.warn('Audio load error for:', currentAudio?.src);
}

function togglePlay() {
    const audio   = currentAudio;
    const player  = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    if (!audio) return;

    if (audio.paused) {
        audio.play().then(() => {
            player?.classList.add('playing');
            if (playBtn) { playBtn.textContent = '⏸'; playBtn.classList.add('playing'); }
        }).catch(console.error);
    } else {
        audio.pause();
        player?.classList.remove('playing');
        if (playBtn) { playBtn.textContent = '▶'; playBtn.classList.remove('playing'); }
    }
}

/* ── Send response ──────────────────────────────────────── */
async function send() {
    if (!canRespond || is_sending) return;
    is_sending = true;

    const s   = stimuli[idx];
    const rt  = (Date.now() - start_time) / 1000;
    const btn = document.getElementById('btn');

    if (btn) { btn.disabled = true; btn.textContent = 'Envoi…'; }

    // Stop audio proprement
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.src = '';
        currentAudio = null;
    }

    const payload = {
        participant_id: participant_id,
        stim_id:        s.stim_id || s.audio_file || String(idx),
        groove:         Number(document.getElementById('g').value),
        complexity:     Number(document.getElementById('c').value),
        rt:             parseFloat(rt.toFixed(3)),
    };

    try {
        const res = await fetch('/response', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

    } catch (e) {
        console.error('Send error:', e);
        is_sending = false;
        if (btn) { btn.disabled = false; btn.textContent = 'Réessayer'; }
        return;
    }

    idx++;

    // Fixation cross
    document.getElementById('content').innerHTML =
        `<div class="fixation">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 550);
}

/* ── Thank you ──────────────────────────────────────────── */
function showThanks() {
    document.getElementById('task').innerHTML = `
        <div class="thanks screen">
            <div class="big">🙏</div>
            <h2>Merci pour votre participation</h2>
            <p style="margin-top:12px;">
                Vos réponses ont été enregistrées.<br>
                Vous pouvez fermer cette page.
            </p>
        </div>
    `;
}

/* ── Helpers ────────────────────────────────────────────── */
function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function preloadOne(s) {
    if (!s?.audio_url) return;
    const a = new Audio();
    a.preload = 'auto';
    a.src = s.audio_url;
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function showError(msg) {
    document.getElementById('app').innerHTML =
        `<div class="card" style="text-align:center;color:var(--muted)">${msg}</div>`;
}

/* ── Boot ───────────────────────────────────────────────── */
init();