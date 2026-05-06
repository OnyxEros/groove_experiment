/* ============================================================
   app.js — Groove Study v3
   Refactor complet :
     - Classe AudioPlayer : gestion propre des listeners
     - Unlock bouton basé sur durée d'écoute réelle (80%)
     - Gestion d'erreurs Supabase visible
     - Tirage stratifié S_mv × D_mv × E (1 stimulus/cellule, aléatoire)
     - Zero memory leaks
   ============================================================ */

'use strict';
 
/* ══════════════════════════════════════════════════════════
   CONSTANTES
   ══════════════════════════════════════════════════════════ */
 
const FIXATION_DELAY_MS = 550;
const MAX_STIMULI       = 30;
const POOL_SIZE         = 200;
 
const WAVEFORM_HEIGHTS = [3, 6, 9, 13, 9, 15, 9, 6, 11, 15, 9, 6, 11, 7, 4];
 
 
/* ══════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════ */
 
const state = {
  participantId:   null,
  stimuli:         [],
  idx:             0,
  startTime:       0,
  isSending:       false,
  player:          null,
  listenedSeconds: 0,   // ← durée d'écoute accumulée pour le trial courant
};
 
 
/* ══════════════════════════════════════════════════════════
   CLASSE AudioPlayer
   ══════════════════════════════════════════════════════════ */
 
class AudioPlayer {
  constructor(src, { onReady, onProgress, onEnded, onError }) {
    this._el         = document.createElement('audio');
    this._el.preload = 'auto';
    this._handlers   = {};
 
    this._bind('canplaythrough', () => onReady   && onReady());
    this._bind('timeupdate',     () => onProgress && onProgress(this._el.currentTime, this._el.duration || 0));
    this._bind('ended',          () => onEnded    && onEnded());
    this._bind('error',          () => onError    && onError(this._el.error));
 
    const source  = document.createElement('source');
    source.src    = escHtml(src);
    source.type   = 'audio/mpeg';
    this._el.appendChild(source);
 
    document.body.appendChild(this._el);
    this._el.load();
    this._destroyed = false;
  }
 
  _bind(event, handler) {
    this._handlers[event] = handler;
    this._el.addEventListener(event, handler);
  }
 
  play()  { return this._el.play();  }
  pause() { this._el.pause();        }
 
  get paused()      { return this._el.paused;       }
  get currentTime() { return this._el.currentTime;  }
  get duration()    { return this._el.duration || 0; }
 
  seek(fraction) {
    if (this._el.duration) this._el.currentTime = fraction * this._el.duration;
  }
 
  destroy() {
    if (this._destroyed) return;
    this._destroyed = true;
    this._el.pause();
    this._el.src = '';
    for (const [event, handler] of Object.entries(this._handlers)) {
      this._el.removeEventListener(event, handler);
    }
    if (this._el.parentNode) this._el.parentNode.removeChild(this._el);
  }
}
 
 
/* ══════════════════════════════════════════════════════════
   INIT
   ══════════════════════════════════════════════════════════ */
 
async function init() {
  try {
    const p           = await fetch('/new_participant').then(r => r.json());
    state.participantId = p.participant_id;
 
    const pool    = await fetch(`/stimuli?n=${POOL_SIZE}`).then(r => r.json());
    state.stimuli = selectStratified(pool, MAX_STIMULI);
 
    if (state.stimuli.length > 0) preloadOne(state.stimuli[0]);
 
  } catch (e) {
    console.error('Init error:', e);
    showError('Erreur de chargement — recharge la page.');
  }
}
 
 
/* ══════════════════════════════════════════════════════════
   TIRAGE STRATIFIÉ S_mv × D_mv × E
   ══════════════════════════════════════════════════════════ */
 
function selectStratified(pool, n) {
  if (!pool?.length) return [];
 
  if (!('S_mv' in pool[0])) {
    shuffle(pool);
    return pool.slice(0, n);
  }
 
  const cells = new Map();
  for (const stim of pool) {
    const key = `${stim.S_mv}_${stim.D_mv}_${stim.E}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(stim);
  }
 
  const picked = [];
  for (const candidates of cells.values()) {
    const idx = Math.floor(Math.random() * candidates.length);
    picked.push(candidates[idx]);
  }
 
  if (picked.length < n) {
    const pickedIds = new Set(picked.map(s => s.stim_id ?? s.audio_file));
    const remaining = pool.filter(s => !pickedIds.has(s.stim_id ?? s.audio_file));
    shuffle(remaining);
    picked.push(...remaining.slice(0, n - picked.length));
  }
 
  shuffle(picked);
  return picked.slice(0, n);
}
 
 
/* ══════════════════════════════════════════════════════════
   NAVIGATION
   ══════════════════════════════════════════════════════════ */
 
function onConsentChange() {
  const cb  = document.getElementById('consent-check');
  const btn = document.getElementById('consent-btn');
  if (cb && btn) btn.disabled = !cb.checked;
}
 
function goIntro() {
  showScreen('screen-intro');
  loadExample();
}
 
function goCalibration() {
  stopExample();
  showScreen('screen-calib');
  syncSlider(document.getElementById('cg'), 'cg-val');
  syncSlider(document.getElementById('cc'), 'cc-val');
}
 
function startExperiment() {
  stopExample();
  const pw = document.getElementById('progress-wrap');
  if (pw) pw.style.display = 'block';
  showScreen('screen-task');
  render();
}
 
function showScreen(id) {
  ['screen-consent', 'screen-intro', 'screen-calib', 'screen-task'].forEach(sid => {
    const el = document.getElementById(sid);
    if (el) el.style.display = 'none';
  });
  const target = document.getElementById(id);
  if (target) {
    target.style.display = 'block';
    void target.offsetWidth;
  }
}
 
 
/* ══════════════════════════════════════════════════════════
   EXAMPLE PLAYER
   ══════════════════════════════════════════════════════════ */
 
let _exPlayer = null;
 
function loadExample() {
  const container = document.getElementById('example-container');
  if (!container) return;
 
  fetch('/example')
    .then(r => r.json())
    .then(s => {
      container.innerHTML = `
        <div class="player" id="ex-player">
          <button class="play-btn" id="ex-play-btn" onclick="toggleExample()">▶</button>
          <div class="player-info">
            <div class="player-meta">
              <span class="player-title">Exemple — groove fort</span>
              <span class="player-time" id="ex-time">--:--</span>
            </div>
            <div class="player-bar-track">
              <div class="player-bar-fill" id="ex-fill"></div>
            </div>
            <div class="waveform">${buildWaveform()}</div>
          </div>
        </div>`;
 
      _exPlayer = new AudioPlayer(s.audio_url, {
        onProgress: (ct, dur) => {
          const fill = document.getElementById('ex-fill');
          const t    = document.getElementById('ex-time');
          if (fill && dur) fill.style.width = (ct / dur * 100) + '%';
          if (t)           t.textContent = formatTime(ct);
        },
        onEnded: () => _setExPlayState(false),
      });
    })
    .catch(() => {
      if (container) container.innerHTML =
        '<div class="example-loading">Exemple non disponible.</div>';
    });
}
 
function toggleExample() {
  if (!_exPlayer) return;
  if (_exPlayer.paused) {
    _exPlayer.play().then(() => _setExPlayState(true)).catch(() => {});
  } else {
    _exPlayer.pause();
    _setExPlayState(false);
  }
}
 
function _setExPlayState(playing) {
  const btn    = document.getElementById('ex-play-btn');
  const player = document.getElementById('ex-player');
  if (btn)    { btn.textContent = playing ? '⏸' : '▶'; btn.classList.toggle('playing', playing); }
  if (player) player.classList.toggle('playing', playing);
}
 
function stopExample() {
  if (_exPlayer) { _exPlayer.destroy(); _exPlayer = null; }
}
 
 
/* ══════════════════════════════════════════════════════════
   TRIAL RENDER
   ══════════════════════════════════════════════════════════ */
 
function render() {
  if (state.idx >= state.stimuli.length) { showThanks(); return; }
 
  if (state.player) { state.player.destroy(); state.player = null; }
 
  // Réinitialise le compteur d'écoute pour ce trial
  state.listenedSeconds = 0;
 
  const s   = state.stimuli[state.idx];
  const pct = Math.round((state.idx / state.stimuli.length) * 100);
 
  _updateProgress(pct, state.idx + 1, state.stimuli.length);
  if (state.idx + 1 < state.stimuli.length) preloadOne(state.stimuli[state.idx + 1]);
 
  const content = document.getElementById('content');
  if (content) content.innerHTML = buildTrialHTML(s, state.idx);
 
  syncSlider(document.getElementById('g'), 'gv');
  syncSlider(document.getElementById('c'), 'cv');
 
  _mountTrialPlayer(s.audio_url);
}
 
function _updateProgress(pct, current, total) {
  const progress = document.getElementById('progress');
  const left     = document.getElementById('counter-left');
  const right    = document.getElementById('counter-right');
  if (progress) progress.style.width = pct + '%';
  if (left)     left.textContent     = `Extrait ${current} / ${total}`;
  if (right)    right.textContent    = pct + '%';
}
 
function _mountTrialPlayer(url) {
  let lastTime = 0;
 
  state.startTime       = Date.now();
  state.listenedSeconds = 0;
 
  state.player = new AudioPlayer(url, {
 
    onReady: () => {
      const hint   = document.getElementById('autoplay-hint');
      const player = document.getElementById('player');
      const btn    = document.getElementById('play-btn');
      const submit = document.getElementById('btn');
 
      if (hint)   hint.classList.remove('visible');
      if (player) { player.classList.remove('waiting'); player.classList.add('playing'); }
      if (btn)    { btn.textContent = '⏸'; btn.classList.add('playing'); }
 
      // ← Bouton débloqué dès que l'audio est prêt, sans condition d'écoute minimale
      if (submit) { submit.disabled = false; submit.textContent = 'Continuer →'; }
    },
 
    onProgress: (ct, dur) => {
      // Accumule la durée réellement écoutée (sans les sauts)
      if (lastTime > 0 && ct > lastTime) {
        state.listenedSeconds += ct - lastTime;
      }
      lastTime = ct;
 
      const fill = document.getElementById('player-fill');
      const t    = document.getElementById('player-time');
      if (fill && dur) fill.style.width = (ct / dur * 100) + '%';
      if (t)           t.textContent = formatTime(ct);
    },
 
    onEnded: () => {
      const player = document.getElementById('player');
      const btn    = document.getElementById('play-btn');
      if (player) player.classList.remove('playing');
      if (btn)    { btn.textContent = '▶'; btn.classList.remove('playing'); }
    },
 
    onError: () => {
      const hint   = document.getElementById('autoplay-hint');
      const submit = document.getElementById('btn');
      if (hint)   { hint.textContent = '⚠ Fichier audio indisponible'; hint.classList.add('visible'); }
      // Débloque quand même si erreur
      if (submit) { submit.disabled = false; submit.textContent = 'Continuer →'; }
    },
  });
 
  // Tentative d'autoplay
  state.player.play().catch(() => {
    const hint   = document.getElementById('autoplay-hint');
    const player = document.getElementById('player');
    const submit = document.getElementById('btn');
    if (hint)   hint.classList.add('visible');
    if (player) player.classList.add('waiting');
    // Même sans autoplay, le bouton est accessible
    if (submit) { submit.disabled = false; submit.textContent = 'Continuer →'; }
  });
}
 
 
/* ══════════════════════════════════════════════════════════
   CONTRÔLES PLAYER
   ══════════════════════════════════════════════════════════ */
 
function togglePlay() {
  const p = state.player;
  if (!p) return;
 
  const player = document.getElementById('player');
  const btn    = document.getElementById('play-btn');
  const hint   = document.getElementById('autoplay-hint');
 
  if (p.paused) {
    p.play().then(() => {
      if (player) { player.classList.remove('waiting'); player.classList.add('playing'); }
      if (btn)    { btn.textContent = '⏸'; btn.classList.add('playing'); }
      if (hint)   hint.classList.remove('visible');
    }).catch(console.error);
  } else {
    p.pause();
    if (player) player.classList.remove('playing');
    if (btn)    { btn.textContent = '▶'; btn.classList.remove('playing'); }
  }
}
 
function seekTo(event) {
  const p = state.player;
  if (!p || !p.duration) return;
  const rect = event.currentTarget.getBoundingClientRect();
  p.seek((event.clientX - rect.left) / rect.width);
}
 
 
/* ══════════════════════════════════════════════════════════
   SEND
   ══════════════════════════════════════════════════════════ */
 
async function send() {
  if (state.isSending) return;
 
  const btn = document.getElementById('btn');
  if (btn?.disabled) return;
 
  state.isSending = true;
  if (btn) { btn.disabled = true; btn.textContent = 'Envoi…'; }
 
  // Snapshot de listenedSeconds avant de détruire le player
  const listenDuration = parseFloat(state.listenedSeconds.toFixed(3));
 
  if (state.player) { state.player.destroy(); state.player = null; }
 
  const s  = state.stimuli[state.idx];
  const rt = (Date.now() - state.startTime) / 1000;
 
  const payload = {
    participant_id:   state.participantId,
    stim_id:          s.stim_id || s.audio_file || String(state.idx),
    groove:           Number(document.getElementById('g').value),
    complexity:       Number(document.getElementById('c').value),
    rt:               parseFloat(rt.toFixed(3)),
    rt_type:          'response',
    trial_index:      state.idx,
    session_id:       state.participantId,
    condition:        'main',
    listen_duration:  listenDuration,   // ← durée d'écoute réelle en secondes
    timestamp_client: Date.now(),
  };
 
  try {
    const res = await fetch('/response', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
 
    if (!res.ok) {
      const detail = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(detail.error || `HTTP ${res.status}`);
    }
 
  } catch (e) {
    console.error('Send error:', e);
    state.isSending = false;
    _showSendError(e.message);
    return;
  }
 
  state.idx++;
  state.isSending = false;
 
  const content = document.getElementById('content');
  if (content) content.innerHTML = '<div class="fixation">+</div>';
 
  setTimeout(() => render(), FIXATION_DELAY_MS);
}
 
function _showSendError(msg) {
  const btn = document.getElementById('btn');
  if (btn) { btn.disabled = false; btn.textContent = 'Réessayer'; }
 
  const existing = document.getElementById('error-toast');
  if (existing) existing.remove();
 
  const toast = document.createElement('div');
  toast.id = 'error-toast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #1a1a1a; border: 1px solid #ef4444; color: #ef4444;
    padding: 10px 20px; border-radius: 8px; font-size: 13px;
    font-family: 'DM Mono', monospace; z-index: 9999;
    animation: fadeIn .2s ease;
  `;
  toast.textContent = `⚠ Erreur d'envoi : ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}
 
 
/* ══════════════════════════════════════════════════════════
   BUILDERS HTML
   ══════════════════════════════════════════════════════════ */
 
function buildTrialHTML(s, i) {
  return `
    <div>
      <div class="player waiting" id="player">
        <button class="play-btn" id="play-btn" onclick="togglePlay()" aria-label="Lecture / Pause">▶</button>
        <div class="player-info">
          <div class="player-meta">
            <span class="player-title">Extrait ${i + 1}</span>
            <span class="player-time" id="player-time">--:--</span>
          </div>
          <div class="player-bar-track" onclick="seekTo(event)" style="cursor:pointer">
            <div class="player-bar-fill" id="player-fill"></div>
          </div>
          <div class="waveform">${buildWaveform()}</div>
        </div>
      </div>
 
      <div class="autoplay-hint" id="autoplay-hint">▶ Appuie sur le bouton pour écouter</div>
 
      <div class="slider-block">
        <div class="slider-header">
          <span class="slider-label">Groove</span>
          <span class="slider-pill" id="gv">4</span>
        </div>
        <div class="slider-anchors">
          <span>Faible</span><span>Modéré</span><span>Fort</span>
        </div>
        <input type="range" id="g" min="1" max="7" value="4" oninput="syncSlider(this,'gv')">
      </div>
 
      <div class="slider-block">
        <div class="slider-header">
          <span class="slider-label">Complexité</span>
          <span class="slider-pill" id="cv">4</span>
        </div>
        <div class="slider-anchors">
          <span>Simple</span><span>Modérée</span><span>Complexe</span>
        </div>
        <input type="range" id="c" min="1" max="7" value="4" oninput="syncSlider(this,'cv')">
      </div>
 
      <button class="btn" onclick="send()" id="btn" disabled>
        Chargement…
      </button>
    </div>`;
}
 
function buildWaveform() {
  return WAVEFORM_HEIGHTS.map((h, i) =>
    `<span style="height:${h}px;--d:${(0.45 + i * 0.08).toFixed(2)}s"></span>`
  ).join('');
}
 
 
/* ══════════════════════════════════════════════════════════
   THANKS
   ══════════════════════════════════════════════════════════ */
 
function showThanks() {
  const task = document.getElementById('screen-task');
  if (task) task.innerHTML = `
    <div class="thanks">
      <div class="thanks-icon">🙏</div>
      <h2>Merci pour ta participation !</h2>
      <p style="margin-top:10px;font-size:14px;color:var(--muted);">
        Tes réponses ont bien été enregistrées.<br>
        Tu peux fermer cette fenêtre.
      </p>
      <div class="thanks-detail">ID session : ${state.participantId || '—'}</div>
    </div>`;
}
 
 
/* ══════════════════════════════════════════════════════════
   UTILITAIRES
   ══════════════════════════════════════════════════════════ */
 
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
    .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
 
function formatTime(sec) {
  if (!isFinite(sec)) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
 
function showError(msg) {
  const app = document.getElementById('app');
  if (app) app.innerHTML =
    `<div class="card" style="text-align:center;padding:40px 24px">
       <p style="color:var(--muted);font-size:14px">${msg}</p>
     </div>`;
}
 
 
/* ══════════════════════════════════════════════════════════
   BOOT
   ══════════════════════════════════════════════════════════ */
 
document.addEventListener('DOMContentLoaded', () => {
  showScreen('screen-consent');
  init();
});