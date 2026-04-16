HTML_PAGE = """
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f0f12;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        #app {
            width: 90%;
            max-width: 500px;
            background: #1b1b22;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        h2 {
            text-align: center;
            margin-bottom: 20px;
        }

        .slider-block {
            margin: 20px 0;
        }

        input[type=range] {
            width: 100%;
        }

        button {
            width: 100%;
            padding: 12px;
            margin-top: 15px;
            border: none;
            border-radius: 10px;
            background: #4c7dff;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:disabled {
            background: #2c3550;
        }

        audio {
            width: 100%;
            margin-top: 10px;
        }

        .counter {
            text-align: center;
            opacity: 0.7;
            margin-bottom: 10px;
        }
    </style>
</head>

<body>
    <div id="app">
        <h2>🎧 Groove Experiment</h2>
        <div class="counter" id="counter"></div>
    </div>

<script>

let participant_id = null;
let stimuli = [];
let idx = 0;
let start_time = 0;
let is_sending = false;

async function init() {

    participant_id = (await fetch("/new_participant").then(r => r.json())).participant_id;

    stimuli = await fetch("/stimuli?n=20").then(r => r.json());

    render();
}

function render() {

    if (idx >= stimuli.length) {
        document.getElementById("app").innerHTML = "<h2>Fin 🙏</h2>";
        return;
    }

    start_time = Date.now();

    let s = stimuli[idx];

    document.getElementById("app").innerHTML = `

        <h2>🎧 Groove Experiment</h2>
        <div class="counter">${idx+1} / ${stimuli.length}</div>

        <audio controls autoplay>
            <source src="${s.audio_url}" type="audio/mp3">
        </audio>

        <div class="slider-block">
            <label>Groove</label>
            <input type="range" id="g" min="1" max="7" value="4">
        </div>

        <div class="slider-block">
            <label>Complexité</label>
            <input type="range" id="c" min="1" max="7" value="4">
        </div>

        <button onclick="send()" id="btn">
            ${is_sending ? "Envoi..." : "Next"}
        </button>
    `;
}

async function send() {

    if (is_sending) return;
    is_sending = true;

    let s = stimuli[idx];

    let rt = (Date.now() - start_time) / 1000;

    document.getElementById("btn").disabled = true;

    await fetch("/response", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            participant_id: participant_id,
            stim_id: s.stim_id || s.audio_file,
            groove: Number(document.getElementById("g").value),
            complexity: Number(document.getElementById("c").value),
            rt: isNaN(rt) ? 0 : rt,
            timestamp_client: Date.now()
        })
    });

    idx++;
    is_sending = false;
    render();
}

init();

</script>
</body>
</html>
"""