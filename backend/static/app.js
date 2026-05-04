/* ============================================================
   app.js — Groove Study v2
   ============================================================ */

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
async function init() {
    try {
        const p = await fetch('/new_participant').then(r => r.json());
        participant_id = p.participant_id;

        stimuli = await fetch('/stimuli?n=20').then(r => r.json());
        shuffle(stimuli);

        if (stimuli.length > 0) preloadOne(stimuli[0]);

    } catch (e) {
        console.error('Init error:', e);
        showError('Erreur de chargement — recharge la page.');
    }
}

/* ── Navigation intro / calibration ────────────────────── */
function goCalibration() {
    showScreen('screen-calib');
    // Initialise le fill des sliders de calibration
    syncSlider(document.getElementById('cg'), 'cg-val');
    syncSlider(document.getElementById('cc'), 'cc-val');
}

function goIntro() {
    showScreen('screen-intro');
}

/* ── Start experiment ───────────────────────────────────── */
function startExperiment() {
    showScreen('screen-task');
    document.getElementById('screen-task').classList.add('screen');
    render();
}

/* ── Screen switcher ────────────────────────────────────── */
function showScreen(id) {
    ['screen-intro', 'screen-calib', 'screen-task'].forEach(s => {
        const el = document.getElementById(s);
        if (el) el.style.display = 'none';
    });
    const target = document.getElementById(id);
    if (target) {
        target.style.display = 'block';
        target.classList.add('screen');
        // Re-trigger animation
        void target.offsetWidth;
    }
}

/* ── Slider sync (fill track + pill value) ──────────────── */
function syncSlider(input, pillId) {
    const min  = Number(input.min);
    const max  = Number(input.max);
    const val  = Number(input.value);
    const pct  = ((val - min) / (max - min)) * 100;

    // Filled track via CSS custom property
    input.style.setProperty('--fill', pct + '%');

    // Pill value
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

    document.getElementById('progress').style.width      = pct + '%';
    document.getElementById('counter-left').textContent  = `Extrait ${idx + 1} / ${stimuli.length}`;
    document.getElementById('counter-right').textContent = pct + '%';

    canRespond = false;

    if (idx + 1 < stimuli.length) preloadOne(stimuli[idx + 1]);

    document.getElementById('content').innerHTML = buildTrialHTML(s, idx);

    mountPlayer(s.audio_url);

    // Init slider fills
    ['g', 'c'].forEach(id => {
        const el = document.getElementById(id);
        if (el) syncSlider(el, id === 'g' ? 'gv' : 'cv');
    });
}

/* ── Trial HTML ─────────────────────────────────────────── */
function buildTrialHTML(s, i) {
    return `
    <div class="screen">
        <!-- Player -->
        <div class="player waiting" id="player">
            <button class="play-btn" id="play-btn" onclick="togglePlay()" aria-label="Lecture / Pause">
                ▶
            </button>
            <div class="player-info">
                <div class="player-meta">
                    <span class="player-title">Extrait ${i + 1}</span>
                    <span class="player-time" id="player-time">--:--</span>
                </div>
                <div class="player-bar-track" onclick="seekTo(event)" style="cursor:pointer">
                    <div class="player-bar-fill" id="player-fill"></div>
                </div>
                <div class="waveform" id="waveform">${buildWaveform()}</div>
            </div>
        </div>

        <div class="autoplay-hint" id="autoplay-hint">
            ▶ Appuie sur le bouton pour écouter
        </div>

        <audio id="audio" preload="auto">
            <source src="${escHtml(s.audio_url)}" type="audio/mpeg">
        </audio>

        <!-- Sliders -->
        <div class="slider-block">
            <div class="slider-header">
                <span class="slider-label">Groove</span>
                <span class="slider-pill" id="gv">4</span>
            </div>
            <div class="slider-anchors">
                <span>Faible</span><span>Modéré</span><span>Fort</span>
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
                <span>Simple</span><span>Modérée</span><span>Complexe</span>
            </div>
            <input type="range" id="c" min="1" max="7" value="4"
                   oninput="syncSlider(this,'cv')">
        </div>

        <button class="btn" onclick="send()" id="btn" disabled>
            Écoute en cours…
        </button>
    </div>`;
}

/* ── Waveform ────────────────────────────────────────────── */
function buildWaveform() {
    const heights = [3, 6, 9, 13, 9, 15, 9, 6, 11, 15, 9, 6, 11, 7, 4];
    return heights.map((h, i) =>
        `<span style="height:${h}px;--d:${(0.45 + i * 0.08).toFixed(2)}s"></span>`
    ).join('');
}

/* ── Player mount ───────────────────────────────────────── */
function mountPlayer(url) {
    const audio = document.getElementById('audio');
    currentAudio = audio;

    audio.addEventListener('canplaythrough', onCanPlay,   { once: true });
    audio.addEventListener('timeupdate',     onTimeUpdate);
    audio.addEventListener('ended',          onEnded);
    audio.addEventListener('error',          onAudioError);

    audio.play().catch(() => {
        // Autoplay bloqué → affiche hint
        const hint = document.getElementById('autoplay-hint');
        if (hint) hint.classList.add('visible');

        const player = document.getElementById('player');
        if (player) player.classList.add('waiting');
    });
}

function onCanPlay() {
    const hint = document.getElementById('autoplay-hint');
    if (hint) hint.classList.remove('visible');

    const player  = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    if (player)  { player.classList.remove('waiting'); player.classList.add('playing'); }
    if (playBtn) { playBtn.textContent = '⏸'; playBtn.classList.add('playing'); }

    start_time = Date.now();

    setTimeout(() => {
        canRespond = true;
        const btn = document.getElementById('btn');
        if (btn) { btn.disabled = false; btn.textContent = 'Continuer →'; }
    }, 800);
}

function onTimeUpdate() {
    const audio = currentAudio;
    if (!audio || !audio.duration) return;

    const pct  = (audio.currentTime / audio.duration) * 100;
    const fill = document.getElementById('player-fill');
    if (fill) fill.style.width = pct + '%';

    const timeEl = document.getElementById('player-time');
    if (timeEl) timeEl.textContent = formatTime(audio.currentTime);
}

function onEnded() {
    const player  = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    if (player)  player.classList.remove('playing');
    if (playBtn) { playBtn.textContent = '▶'; playBtn.classList.remove('playing'); }
}

function onAudioError() {
    console.warn('Audio error:', currentAudio?.src);
    canRespond = true;
    const btn = document.getElementById('btn');
    if (btn) { btn.disabled = false; btn.textContent = 'Continuer →'; }
    const hint = document.getElementById('autoplay-hint');
    if (hint) { hint.textContent = '⚠ Fichier audio indisponible'; hint.classList.add('visible'); }
}

function togglePlay() {
    const audio   = currentAudio;
    const player  = document.getElementById('player');
    const playBtn = document.getElementById('play-btn');
    const hint    = document.getElementById('autoplay-hint');
    if (!audio) return;

    if (audio.paused) {
        audio.play().then(() => {
            player?.classList.remove('waiting');
            player?.classList.add('playing');
            if (playBtn) { playBtn.textContent = '⏸'; playBtn.classList.add('playing'); }
            if (hint)    hint.classList.remove('visible');

            // Active le bouton si l'utilisateur a joué manuellement
            if (!canRespond) {
                start_time = Date.now();
                setTimeout(() => {
                    canRespond = true;
                    const btn = document.getElementById('btn');
                    if (btn) { btn.disabled = false; btn.textContent = 'Continuer →'; }
                }, 800);
            }
        }).catch(console.error);
    } else {
        audio.pause();
        player?.classList.remove('playing');
        if (playBtn) { playBtn.textContent = '▶'; playBtn.classList.remove('playing'); }
    }
}

/* Seek en cliquant sur la barre de progression */
function seekTo(event) {
    const audio = currentAudio;
    if (!audio || !audio.duration) return;
    const track = event.currentTarget;
    const rect  = track.getBoundingClientRect();
    const pct   = (event.clientX - rect.left) / rect.width;
    audio.currentTime = pct * audio.duration;
}

/* ── Send ────────────────────────────────────────────────── */
async function send() {
    if (!canRespond || is_sending) return;
    is_sending = true;

    const s   = stimuli[idx];
    const rt  = (Date.now() - start_time) / 1000;
    const btn = document.getElementById('btn');

    if (btn) { btn.disabled = true; btn.textContent = 'Envoi…'; }

    // Stop audio
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

    document.getElementById('content').innerHTML =
        `<div class="fixation">+</div>`;

    setTimeout(() => {
        is_sending = false;
        render();
    }, 550);
}

/* ── Thank you ──────────────────────────────────────────── */
function showThanks() {
    document.getElementById('screen-task').innerHTML = `
        <div class="thanks screen">
            <div class="thanks-icon">🙏</div>
            <h2>Merci pour votre participation</h2>
            <p class="sub" style="margin-top:10px;">
                Vos réponses ont bien été enregistrées.<br>
                Vous pouvez fermer cette fenêtre.
            </p>
            <div class="thanks-detail">ID session : ${participant_id || '—'}</div>
        </div>
    `;
}

/* ── Helpers ─────────────────────────────────────────────── */
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

function formatTime(sec) {
    if (!isFinite(sec)) return '--:--';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function showError(msg) {
    document.getElementById('app').innerHTML =
        `<div class="card" style="text-align:center;padding:40px 24px">
            <p style="color:var(--muted);font-size:14px">${msg}</p>
         </div>`;
}

/* ── Boot ───────────────────────────────────────────────── */
init();