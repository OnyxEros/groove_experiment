HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Groove Study</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & tokens ──────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #080910;
    --surface:    #0f1116;
    --surface2:   #161820;
    --border:     rgba(255,255,255,0.07);
    --accent:     #5c7cff;
    --accent-dim: rgba(92,124,255,0.15);
    --text:       #f0f0f4;
    --muted:      rgba(240,240,244,0.45);
    --faint:      rgba(240,240,244,0.18);
    --success:    #3ecf8e;
    --radius:     14px;
    --mono:       'DM Mono', monospace;
    --sans:       'DM Sans', sans-serif;
}

body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100dvh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    /* Subtle grid noise */
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(92,124,255,0.08), transparent),
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 100% 100%, 32px 32px, 32px 32px;
}

/* ── Shell ───────────────────────────────────────────── */
#app {
    width: 100%;
    max-width: 480px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 28px;
    position: relative;
    overflow: hidden;
}

/* thin top accent line */
#app::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.6;
}

/* ── Typography ──────────────────────────────────────── */
h2 {
    font-family: var(--mono);
    font-weight: 400;
    font-size: 18px;
    letter-spacing: -0.01em;
    text-align: center;
    margin-bottom: 6px;
}

.eyebrow {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    text-align: center;
    margin-bottom: 20px;
}

p, .body-text {
    font-size: 14px;
    line-height: 1.65;
    color: var(--muted);
}

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--muted);
    margin-bottom: 18px;
}

.card b { color: var(--text); font-weight: 500; }

/* ── Button ──────────────────────────────────────────── */
.btn {
    display: block;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    font-family: var(--sans);
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    margin-top: 20px;
    letter-spacing: 0.01em;
}

.btn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
.btn:active:not(:disabled) { transform: translateY(0); }
.btn:disabled {
    background: var(--surface2);
    color: var(--faint);
    border: 1px solid var(--border);
    cursor: not-allowed;
}

/* ── Progress ────────────────────────────────────────── */
.progress-wrap {
    margin-bottom: 20px;
}

.progress-track {
    height: 3px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
    border-radius: 99px;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--faint);
    margin-top: 7px;
    letter-spacing: 0.04em;
}

/* ── Audio player (custom) ───────────────────────────── */
.player {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

/* animated waveform bars when playing */
.player::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, var(--accent-dim), transparent 60%);
    opacity: 0;
    transition: opacity 0.4s;
    pointer-events: none;
}

.player.playing::after { opacity: 1; }

.play-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 1.5px solid var(--accent);
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 15px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
    position: relative;
    z-index: 1;
}

.play-btn:hover { background: rgba(92,124,255,0.25); transform: scale(1.05); }
.play-btn.playing { background: var(--accent); color: #fff; }

.player-info {
    flex: 1;
    min-width: 0;
    position: relative;
    z-index: 1;
}

.player-title {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.player-bar-track {
    height: 3px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}

.player-bar-fill {
    height: 100%;
    background: var(--accent);
    width: 0%;
    transition: width 0.25s linear;
    border-radius: 99px;
}

.waveform {
    display: flex;
    align-items: center;
    gap: 2px;
    height: 18px;
    margin-top: 7px;
}

.waveform span {
    display: block;
    width: 2px;
    background: var(--faint);
    border-radius: 1px;
    transition: background 0.3s;
}

.player.playing .waveform span {
    background: var(--accent);
    animation: wave var(--d, 0.8s) ease-in-out infinite alternate;
}

@keyframes wave {
    from { transform: scaleY(0.3); }
    to   { transform: scaleY(1); }
}

audio { display: none; }

/* ── Sliders ─────────────────────────────────────────── */
.slider-block { margin: 20px 0; }

.slider-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}

.slider-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
}

.slider-value {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
    min-width: 16px;
    text-align: right;
}

.scale-labels {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--faint);
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: var(--border);
    border-radius: 99px;
    outline: none;
    cursor: pointer;
}

input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--bg);
    box-shadow: 0 0 0 0 rgba(92,124,255,0.4);
    transition: box-shadow 0.2s;
    cursor: pointer;
}

input[type=range]:focus::-webkit-slider-thumb,
input[type=range]:hover::-webkit-slider-thumb {
    box-shadow: 0 0 0 5px rgba(92,124,255,0.2);
}

input[type=range]::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--bg);
    cursor: pointer;
}

/* ── Fixation cross ──────────────────────────────────── */
.fixation {
    text-align: center;
    padding: 40px 0;
    font-size: 28px;
    color: var(--faint);
    animation: pulse-fix 0.5s ease-in-out;
}

@keyframes pulse-fix {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Screen transitions ──────────────────────────────── */
.screen {
    animation: fadein 0.3s ease;
}

@keyframes fadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Divider ─────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 22px 0;
}

/* ── Thank you ───────────────────────────────────────── */
.thanks {
    text-align: center;
    padding: 20px 0;
}

.thanks .big { font-size: 36px; margin-bottom: 12px; }
</style>
</head>

<body>
<div id="app">

  <!-- ── INTRO ────────────────────────────────────────── -->
  <div id="intro" class="screen">
    <div class="eyebrow">Expérience de perception</div>
    <h2>Groove Study</h2>

    <hr class="divider">

    <div class="card">
      Vous allez écouter <b>20 extraits</b> musicaux courts.<br><br>
      Pour chacun, évaluez deux dimensions :
      <br><br>
      <b>Groove</b> &mdash; l'envie de bouger (taper du pied, hocher la tête).<br>
      <b>Complexité</b> &mdash; la richesse ou simplicité du rythme perçu.
      <br><br>
      Répondez de façon <b>spontanée</b> — il n'y a pas de bonne réponse.
    </div>

    <p style="text-align:center; font-size:12px; color:var(--faint); margin-bottom:4px;">
      Durée estimée : 8–12 minutes
    </p>

    <button class="btn" onclick="startExperiment()">Commencer &rarr;</button>
  </div>

  <!-- ── TASK ─────────────────────────────────────────── -->
  <div id="task" style="display:none;">
    <div class="progress-wrap">
      <div class="progress-track">
        <div class="progress-fill" id="progress"></div>
      </div>
      <div class="progress-label">
        <span id="counter-left">Extrait 1 / 20</span>
        <span id="counter-right">0%</span>
      </div>
    </div>
    <div id="content"></div>
  </div>

</div>

<script src="/static/app.js"></script>
</body>
</html>
"""
