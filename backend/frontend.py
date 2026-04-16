HTML_PAGE = """
<html>
<body>
    <h2>🎧 Groove Experiment</h2>
    <div id="app"></div>

    <script>

    let participant_id = null;
    let stimuli = [];
    let idx = 0;
    let start_time = 0;

    async function init() {

        participant_id = (await fetch("/new_participant").then(r => 
r.json())).participant_id;

        stimuli = await fetch("/stimuli?n=20").then(r => r.json());

        render();
    }

    function render() {

        if (idx >= stimuli.length) {
            document.getElementById("app").innerHTML = "<h3>Fin 🙏</h3>";
            return;
        }

        start_time = Date.now();

        let s = stimuli[idx];

        document.getElementById("app").innerHTML = `
            <h3>${idx+1} / ${stimuli.length}</h3>

            <audio controls autoplay>
                <source src="${s.audio_url}" type="audio/mp3">
            </audio>

            <br><br>

            Groove:
            <input type="range" id="g" min="1" max="7" value="4">

            <br><br>

            Complexité:
            <input type="range" id="c" min="1" max="7" value="4">

            <br><br>

            <button onclick="send()">Next</button>
        `;
    }

    async function send() {

        let s = stimuli[idx];

        await fetch("/response", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                participant_id: participant_id,
                stim_id: s.id,
                groove: parseInt(document.getElementById("g").value),
                complexity: parseInt(document.getElementById("c").value),
                rt: (Date.now() - start_time) / 1000
            })
        });

        idx++;
        render();
    }

    init();

    </script>
</body>
</html>
"""
