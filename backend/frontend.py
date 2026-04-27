HTML_PAGE = r"""
<!DOCTYPE html>
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
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
Arial;
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
}

.subtitle {
    text-align:center;
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
}

button {
    width:100%;
    padding:14px;
    border:none;
    border-radius:12px;
    background: linear-gradient(135deg,#6c8cff,#4c7dff);
    color:white;
    font-size:16px;
    cursor:pointer;
}

.progress { height:6px; background:rgba(255,255,255,0.08); 
border-radius:10px; margin-bottom:18px; }
.progress-bar { height:100%; background:var(--accent); width:0%; }

.slider-block { margin:18px 0; }

.scale-labels {
    display:flex;
    justify-content:space-between;
    font-size:11px;
    color:rgba(255,255,255,0.5);
}
</style>
</head>

<body>

<div id="app">

<div id="intro">
    <h2>🎧 Groove Study</h2>
    <div class="subtitle">Expérience de perception musicale</div>

    <p>
        Évaluez le <b>groove</b> (envie de bouger) et la 
<b>complexité</b>.
    </p>

    <button onclick="start()">Commencer</button>
</div>

<div id="task" style="display:none;">
    <div class="progress"><div class="progress-bar" 
id="progress"></div></div>
    <div id="counter" class="subtitle"></div>
    <div id="content"></div>
</div>

</div>

<script src="/static/app.js"></script>

</body>
</html>
"""
