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

const LISTEN_THRESHOLD  = 0.80;   // fraction de la durée pour débloquer
const FIXATION_DELAY_MS = 550;    // pause inter-essais
const MAX_STIMULI       = 30;     // cap de sécurité (le stratifié peut en produire moins)
const POOL_SIZE         = 200;    // taille du pool à fetcher depuis l'API

const WAVEFORM_HEIGHTS = [3, 6, 9, 13, 9, 15, 9, 6, 11, 15, 9, 6, 11, 7, 4];


/* ══════════════════════════════════════════════════════════
   STATE (un seul objet mutable, jamais de globaux épars)
   ══════════════════════════════════════════════════════════ */

const state = {
  participantId: null,
  stimuli:       [],
  idx:           0,
  startTime:     0,
  isSending:     false,
  player:        null,   // instance AudioPlayer courante
};


/* ══════════════════════════════════════════════════════════
   CLASSE AudioPlayer
   Encapsule un élément <audio> et ses listeners.
   destroy() retire tout proprement.
   ══════════════════════════════════════════════════════════ */

class AudioPlayer {
  /**
   * @param {string}   src          URL de l'audio
   * @param {Function} onReady      appelé quand l'audio peut jouer
   * @param {Function} onProgress   appelé à chaque timeupdate (currentTime, duration)
   * @param {Function} onEnded      appelé à la fin de la lecture
   * @param {Function} onError      appelé en cas d'erreur
   */
  constructor(src, { onReady, onProgress, onEnded, onError }) {
    this._el       = document.createElement('audio');
    this._el.preload = 'auto';

    this._handlers = {};   // stocke les refs pour removeEventListener

    this._bind('canplaythrough', () => onReady && onReady());
    this._bind('timeupdate',     () => onProgress && onProgress(this._el.currentTime, this._el.duration || 0));
    this._bind('ended',          () => onEnded && onEnded());
    this._bind('error',          () => onError && onError(this._el.error));

    // Ajouter la source APRÈS les listeners (évite race condition Safari)
    const source = document.createElement('source');
    source.src  = escHtml(src);
    source.type = 'audio/mpeg';
    this._el.appendChild(source);

    document.body.appendChild(this._el);   // nécessaire sur certains navigateurs
    this._el.load();

    this._destroyed = false;
  }

  _bind(event, handler) {
    this._handlers[event] = handler;
    this._el.addEventListener(event, handler);
  }

  play()  { return this._el.play();  }
  pause() { this._el.pause();        }

  get paused()       { return this._el.paused;       }
  get currentTime()  { return this._el.currentTime;  }
  get duration()     { return this._el.duration || 0; }

  seek(fraction) {
    if (this._el.duration) {
      this._el.currentTime = fraction * this._el.duration;
    }
  }

  /** Retire tous les listeners et détache l'élément du DOM. */
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
    const p = await fetch('/new_participant').then(r => r.json());
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
   ──────────────────────────────────────────────────────────
   Pour chaque cellule du design factoriel, on tire AU HASARD
   un stimulus parmi tous ceux disponibles dans cette cellule.

   Pourquoi stratifié plutôt que maximin ?
     - Garantit un effectif équilibré par condition (S_mv, D_mv, E)
       → les Kruskal-Wallis et les stats par condition ont des groupes
          de taille comparable, pas biaisés par la sélection
     - Chaque participant voit une couverture uniforme du design space
     - La variance inter-participants est réelle : deux sessions
       consécutives ne voient pas exactement les mêmes stimuli
     - Compatible avec l'ICC : tous les participants couvrent les
       mêmes cellules, mais pas forcément le même stim dans chaque cellule

   Fallback : si les métadonnées S_mv/D_mv/E sont absentes du pool,
   on fait un shuffle + slice classique.
   ══════════════════════════════════════════════════════════ */

function selectStratified(pool, n) {
  if (!pool?.length) return [];

  // Fallback si les colonnes de design sont absentes
  if (!('S_mv' in pool[0])) {
    shuffle(pool);
    return pool.slice(0, n);
  }

  // ── Groupement par cellule S_mv × D_mv × E ──────────────
  const cells = new Map();

  for (const stim of pool) {
    const key = `${stim.S_mv}_${stim.D_mv}_${stim.E}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(stim);
  }

  // ── Tirage aléatoire d'un stimulus par cellule ──────────
  const picked = [];

  for (const candidates of cells.values()) {
    // Fisher-Yates sur les candidats de la cellule, on prend le premier
    const idx = Math.floor(Math.random() * candidates.length);
    picked.push(candidates[idx]);
  }

  // ── Si trop peu de cellules, compléter avec des tirages aléatoires
  //    parmi les stimuli non encore sélectionnés ─────────────
  if (picked.length < n) {
    const pickedIds = new Set(picked.map(s => s.stim_id ?? s.audio_file));
    const remaining = pool.filter(s => !pickedIds.has(s.stim_id ?? s.audio_file));
    shuffle(remaining);
    picked.push(...remaining.slice(0, n - picked.length));
  }

  // ── Ordre aléatoire pour chaque session ─────────────────
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
   EXAMPLE PLAYER (screen-intro)
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
  if (_exPlayer) {
    _exPlayer.destroy();
    _exPlayer = null;
  }
}


/* ══════════════════════════════════════════════════════════
   TRIAL RENDER
   ══════════════════════════════════════════════════════════ */

function render() {
  if (state.idx >= state.stimuli.length) { showThanks(); return; }

  // Détruire le player précédent proprement
  if (state.player) {
    state.player.destroy();
    state.player = null;
  }

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
  if (left)     left.textContent  = `Extrait ${current} / ${total}`;
  if (right)    right.textContent = pct + '%';
}

function _mountTrialPlayer(url) {
  let listenedSeconds = 0;
  let lastTime        = 0;
  let unlocked        = false;

  state.startTime = Date.now();

  state.player = new AudioPlayer(url, {

    onReady: () => {
      const hint   = document.getElementById('autoplay-hint');
      const player = document.getElementById('player');
      const btn    = document.getElementById('play-btn');
      if (hint)   hint.classList.remove('visible');
      if (player) { player.classList.remove('waiting'); player.classList.add('playing'); }
      if (btn)    { btn.textContent = '⏸'; btn.classList.add('playing'); }
    },

    onProgress: (ct, dur) => {
      // Accumule uniquement le temps réellement écouté
      if (lastTime > 0 && ct > lastTime) {
        listenedSeconds += ct - lastTime;
      }
      lastTime = ct;

      // Mise à jour barre de progression
      const fill = document.getElementById('player-fill');
      const t    = document.getElementById('player-time');
      if (fill && dur) fill.style.width = (ct / dur * 100) + '%';
      if (t)           t.textContent = formatTime(ct);

      // Déblocage basé sur écoute réelle
      if (!unlocked && dur > 0 && listenedSeconds >= dur * LISTEN_THRESHOLD) {
        unlocked = true;
        _unlockButton();
      }
    },

    onEnded: () => {
      const player = document.getElementById('player');
      const btn    = document.getElementById('play-btn');
      if (player) player.classList.remove('playing');
      if (btn)    { btn.textContent = '▶'; btn.classList.remove('playing'); }

      // Si l'utilisateur est arrivé à la fin sans avoir atteint le seuil
      if (!unlocked) {
        unlocked = true;
        _unlockButton();
      }
    },

    onError: () => {
      const hint = document.getElementById('autoplay-hint');
      if (hint) { hint.textContent = '⚠ Fichier audio indisponible'; hint.classList.add('visible'); }
      // Débloque quand même pour ne pas bloquer le participant
      if (!unlocked) { unlocked = true; _unlockButton(); }
    },
  });

  // Tentative d'autoplay
  state.player.play().catch(() => {
    const hint   = document.getElementById('autoplay-hint');
    const player = document.getElementById('player');
    if (hint)   hint.classList.add('visible');
    if (player) player.classList.add('waiting');
  });
}

function _unlockButton() {
  const btn = document.getElementById('btn');
  if (btn) { btn.disabled = false; btn.textContent = 'Continuer →'; }
}


/* ══════════════════════════════════════════════════════════
   CONTRÔLES PLAYER (appelés depuis le HTML inline)
   ══════════════════════════════════════════════════════════ */

function togglePlay() {
  const p = state.player;
  if (!p) return;

  const player  = document.getElementById('player');
  const btn     = document.getElementById('play-btn');
  const hint    = document.getElementById('autoplay-hint');

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
  if (btn?.disabled) return;   // bouton non encore débloqué

  state.isSending = true;
  if (btn) { btn.disabled = true; btn.textContent = 'Envoi…'; }

  // Détruire le player avant la requête
  if (state.player) {
    state.player.destroy();
    state.player = null;
  }

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

  // Fixation cross
  const content = document.getElementById('content');
  if (content) content.innerHTML = '<div class="fixation">+</div>';

  setTimeout(() => render(), FIXATION_DELAY_MS);
}

function _showSendError(msg) {
  const btn = document.getElementById('btn');
  if (btn) { btn.disabled = false; btn.textContent = 'Réessayer'; }

  // Toast d'erreur non-intrusif
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
        Écoute en cours…
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

/** Précharge silencieusement l'audio du prochain stimulus. */
function preloadOne(s) {
  if (!s?.audio_url) return;
  const a = new Audio();
  a.preload = 'auto';
  a.src = s.audio_url;
  // L'élément reste orphelin — c'est voulu : le navigateur le met en cache
  // et le GC le récupère quand il n'est plus référencé.
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