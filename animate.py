import json
import math
import random
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from multiprocessing import Pool, cpu_count

# Import the greedy LL parallel trie engine directly
from strategies.sprt_greedyLL_parallel_trie import (
    build_trie, calculate_true_feedback, apply_noise,
    parallel_trie_update, THRESHOLD
)

def get_game_history(target_word, dictionary, p_correct=0.95, p_wrong=0.025, max_turns=100):
    """
    Runs an autonomous game using the Greedy LL strategy, 
    but quietly logs every state metric at each turn for animation.
    """
    trie_root = build_trie(dictionary)
    word_lls = {w: 0.0 for w in dictionary}
    history = []
    
    cores = cpu_count()
    with Pool(processes=cores) as pool:
        # Save T=0 initial state
        ranked = sorted(word_lls.items(), key=lambda x: x[1], reverse=True)
        top_n = [{"word": w, "ll": round(ll, 2)} for w, ll in ranked[:10]]
        history.append({
            "turn": 0,
            "guess": "",
            "true_fb": [],
            "obs_fb": [],
            "top_contenders": top_n
        })
        
        for turn in range(1, max_turns + 1):
            # Select the greedily optimal token
            guess = max(word_lls, key=word_lls.get) 
            
            # Run Physics
            true_fb = calculate_true_feedback(guess, target_word)
            obs_fb = apply_noise(true_fb, p_correct, p_wrong)
            
            # Map-Reduce LLR bounded updates across multicores
            exact_jumps = parallel_trie_update(trie_root, guess, obs_fb, pool, p_correct, p_wrong)
            for w, jump in exact_jumps.items():
                word_lls[w] += jump
                
            # Log exact tracking metric states post-update
            ranked = sorted(word_lls.items(), key=lambda x: x[1], reverse=True)
            top_n = [{"word": w, "ll": round(ll, 2)} for w, ll in ranked[:10]]
            
            history.append({
                "turn": turn,
                "guess": guess,
                "true_fb": list(true_fb),
                "obs_fb": list(obs_fb),
                "top_contenders": top_n
            })
            
            if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1] >= THRESHOLD):
                break
                
    return history

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wordle Solver Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            color: #333;
            line-height: 1.6;
            margin: 0;
            padding: 40px;
            display: flex;
            justify-content: center;
        }
        .container {
            background: #fff;
            max-width: 700px;
            width: 100%;
            padding: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        h1 {
            font-size: 1.5em;
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            color: #222;
        }
        .input-bar {
            background: #f4f6f8;
            padding: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .input-bar input[type="text"] {
            padding: 8px;
            font-size: 1em;
            border: 1px solid #ccc;
            border-radius: 4px;
            flex-grow: 1;
            min-width: 140px;
        }
        .noise-group {
            display: flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        }
        .noise-group label {
            font-weight: bold;
            color: #475569;
            font-size: 0.95em;
        }
        .noise-group input[type="number"] {
            width: 60px;
            padding: 8px 6px;
            font-size: 1em;
            border: 1px solid #ccc;
            border-radius: 4px;
            text-align: center;
        }
        .noise-hint {
            font-size: 0.78em;
            color: #64748b;
            font-style: italic;
        }
        .input-bar button {
            background: #4a90e2;
            color: #fff;
            border: none;
            padding: 9px 18px;
            font-size: 1.05em;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .input-bar button:hover:not(:disabled) {
            background: #357abd;
        }
        .input-bar button:disabled {
            background: #a0c4ec;
            cursor: not-allowed;
        }
        .meta-info {
            margin-bottom: 20px;
            font-size: 1.1em;
        }
        .turn-counter {
            font-weight: bold;
            color: #555;
        }
        .board {
            display: flex;
            gap: 40px;
            margin-bottom: 30px;
        }
        .column {
            display: flex;
            flex-direction: column;
        }
        .column-header {
            font-size: 0.85em;
            text-transform: uppercase;
            color: #777;
            margin-bottom: 5px;
        }
        .tiles {
            display: flex;
            gap: 5px;
        }
        .tile {
            width: 40px;
            height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid #ccc;
            background: #fafafa;
            font-weight: bold;
            font-size: 1.2em;
            text-transform: uppercase;
        }
        .color-0 { background-color: #e0e0e0; color: #333; border-color: #ccc; }
        .color-1 { background-color: #ffeb3b; color: #333; border-color: #fbc02d; }
        .color-2 { background-color: #4caf50; color: #fff; border-color: #388e3c; }
        
        .contender-list {
            position: relative;
            height: 250px;
            border: 1px solid #eee;
            border-radius: 4px;
            padding: 10px;
            background: #fafafa;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .contender-row {
            position: absolute;
            left: 10px;
            right: 10px;
            height: 40px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            transition: transform 0.4s ease, opacity 0.4s ease;
        }
        .c-word {
            font-family: monospace;
            font-size: 1.1em;
            font-weight: bold;
            width: 80px;
            text-transform: uppercase;
        }
        .c-bar-container {
            flex-grow: 1;
            height: 8px;
            background: #eee;
            margin: 0 15px;
            border-radius: 4px;
            overflow: hidden;
        }
        .c-bar {
            height: 100%;
            background: #4a90e2;
            width: 0%;
            transition: width 0.4s ease;
        }
        .c-ll {
            width: 50px;
            text-align: right;
            font-family: monospace;
            color: #555;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .nav-btn {
            padding: 8px 16px;
            font-size: 1em;
            cursor: pointer;
            border: 1px solid #ccc;
            background: #fff;
            border-radius: 4px;
            color: #333;
        }
        .nav-btn:hover:not(:disabled) {
            background: #f0f0f0;
        }
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .end-state {
            display: none;
            padding: 15px;
            background: #e8f5e9;
            border: 1px solid #c8e6c9;
            border-radius: 4px;
            text-align: center;
        }
        .end-state p { margin: 0 0 10px 0; font-size: 1.1em; }
        .end-state button {
            padding: 6px 12px;
            margin: 0 5px;
            cursor: pointer;
            border: 1px solid #81c784;
            background: #fff;
            border-radius: 4px;
            color: #333;
        }
        .end-state button:hover { background: #f1f8e9; }
    </style>
</head>
<body>

<div class="container">
    <h1>Wordle Solver Analysis</h1>
    
    <div class="input-bar">
        <label style="font-weight: bold; color: #475569;">Target Word:</label>
        <input type="text" id="targetInput" placeholder="Enter a 5 or 6 letter word... (e.g. BREAD)" />
        <div class="noise-group">
            <label for="noisePct">Noise %:</label>
            <input type="number" id="noisePct" value="10" min="0" max="49" step="1" title="Noise level" />
            <span class="noise-hint"></span>
        </div>
        <button id="runBtn" onclick="runSimulation()">Solve!</button>
        <span id="loading" style="display:none; color: #64748b; font-style: italic; font-size: 0.9em;">Computing bounds...</span>
    </div>

    <div id="app-body" style="display: none;">
        <div class="meta-info">
            <div>Target Evaluated: <strong id="ui-target-word">...</strong></div>
            <div class="turn-counter">Turn Progress: <span id="turn-num">0</span> / <span id="max-num">0</span></div>
        </div>

        <div class="board">
            <div class="column">
                <div class="column-header">Current Guess</div>
                <div class="tiles" id="guess-tiles"></div>
            </div>
            <div class="column">
                <div class="column-header">Noisy Feedback</div>
                <div class="tiles" id="sensor-tiles"></div>
            </div>
        </div>

        <div class="column-header">Top 5 Candidates (Log-Likelihood Rankings)</div>
        <div class="contender-list" id="contender-list"></div>

        <div class="controls">
            <button class="nav-btn" id="prev-btn" onclick="goPrev()">Previous Turn</button>
            <button class="nav-btn" id="next-btn" onclick="goNext()">Next Turn</button>
        </div>

        <div class="end-state" id="end-state">
            <p>The greedy algorithm has finalized its bounds and believes the word is <strong id="final-guess">...</strong>.</p>
            <p>Is this correct?</p>
            <button onclick="alert('Great! The mathematical threshold converged accurately.')">Yes</button>
            <button onclick="alert('Sorry! The algorithm might have converged into another local minima because of the noisy feedback.')">No</button>
        </div>
    </div>
</div>

<script>
    let historyData = [];
    let currentTurn = 0;
    let totalTurns = 0;

    const container = document.getElementById('contender-list');
    let elements = {};

    function runSimulation() {
        const word = document.getElementById('targetInput').value.trim().toLowerCase();
        if(!word) return;

        document.getElementById('runBtn').disabled = true;
        document.getElementById('loading').style.display = 'inline';
        document.getElementById('app-body').style.display = 'none';

        const noisePct = parseFloat(document.getElementById('noisePct').value) || 0;
        fetch('/simulate', {
            method: 'POST',
            body: JSON.stringify({word: word, noise_pct: noisePct})
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById('runBtn').disabled = false;
            document.getElementById('loading').style.display = 'none';
            if(data.error) {
                alert("Backend Error: " + data.error);
                return;
            }
            historyData = data.history;
            totalTurns = historyData.length - 1;
            document.getElementById('ui-target-word').innerText = word.toUpperCase();
            document.getElementById('max-num').innerText = totalTurns;
            document.getElementById('app-body').style.display = 'block';
            currentTurn = 0;
            renderState(0);
        })
        .catch(err => {
            alert("Error communicating with backend simulator.");
            document.getElementById('runBtn').disabled = false;
            document.getElementById('loading').style.display = 'none';
        });
    }

    function getMaxLL(turnData) { return Math.max(1, ...turnData.top_contenders.map(c => c.ll)); }
    function getMinLL(turnData) { return Math.min(0, ...turnData.top_contenders.map(c => c.ll)); }

    function renderState(turnIdx) {
        const state = historyData[turnIdx];
        document.getElementById('turn-num').innerText = state.turn;

        const guessContainer = document.getElementById('guess-tiles');
        const sensorContainer = document.getElementById('sensor-tiles');
        guessContainer.innerHTML = '';
        sensorContainer.innerHTML = '';

        if(state.turn === 0) {
            const length = historyData[historyData.length - 1].guess.length || 5;
            for(let i=0; i<length; i++) {
                guessContainer.innerHTML += `<div class="tile">?</div>`;
                sensorContainer.innerHTML += `<div class="tile"></div>`;
            }
        } else {
            for(let i=0; i<state.guess.length; i++) {
                guessContainer.innerHTML += `<div class="tile">${state.guess[i].toUpperCase()}</div>`;
                sensorContainer.innerHTML += `<div class="tile color-${state.obs_fb[i]}"></div>`;
            }
        }

        const maxLL = getMaxLL(state);
        let minLL = getMinLL(state);
        if (minLL < -50) minLL = -50; 
        const llRange = maxLL - minLL + 1;

        Object.keys(elements).forEach(w => elements[w].dataset.active = "false");

        for(let i=0; i<Math.min(5, state.top_contenders.length); i++) {
            const item = state.top_contenders[i];
            const word = item.word;
            
            if(!elements[word]) {
                const div = document.createElement('div');
                div.className = 'contender-row';
                div.innerHTML = `
                    <div class="c-word">${word}</div>
                    <div class="c-bar-container"><div class="c-bar"></div></div>
                    <div class="c-ll">0.0</div>
                `;
                div.style.transform = `translateY(250px)`;
                div.style.opacity = '0';
                container.appendChild(div);
                elements[word] = div;
                void div.offsetWidth; 
            }

            const el = elements[word];
            el.dataset.active = "true";
            
            el.style.transform = `translateY(${i * 48}px)`;
            el.style.opacity = '1';
            el.style.zIndex = (100 - i).toString();
            
            let rawWidth = ((item.ll - minLL) / llRange) * 100;
            let barWidth = Math.max(0, Math.min(100, rawWidth));
            el.querySelector('.c-bar').style.width = barWidth + '%';
            el.querySelector('.c-ll').innerText = item.ll.toFixed(1);
        }

        Object.keys(elements).forEach(w => {
            if(elements[w].dataset.active === "false") {
                elements[w].style.opacity = '0';
                elements[w].style.transform = `translateY(250px)`;
            }
        });

        document.getElementById('prev-btn').disabled = (currentTurn === 0);
        document.getElementById('next-btn').disabled = (currentTurn === totalTurns);

        const endState = document.getElementById('end-state');
        if (currentTurn === totalTurns && totalTurns > 0) {
            document.getElementById('final-guess').innerText = state.top_contenders[0].word.toUpperCase();
            endState.style.display = 'block';
        } else {
            endState.style.display = 'none';
        }
    }

    function goNext() {
        if(currentTurn < totalTurns) {
            currentTurn++;
            renderState(currentTurn);
        }
    }

    function goPrev() {
        if(currentTurn > 0) {
            currentTurn--;
            renderState(currentTurn);
        }
    }

    window.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight') goNext();
        if (e.key === 'ArrowLeft') goPrev();
    });
</script>
</body>
</html>
"""

class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/simulate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data)
            target = req.get('word', '').lower()
            
            length = len(target)
            dataset_file = f"datasets/english{length}.json"
            
            if not os.path.exists(dataset_file):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                err = json.dumps({"error": f"Internal Error: Dataset {dataset_file} not found for {length}-letter words."})
                self.wfile.write(err.encode('utf-8'))
                return
                
            with open(dataset_file, "r") as f:
                dictionary = json.load(f)
                
            if target not in dictionary:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                err = json.dumps({"error": f"The word '{target}' does not exist inside {dataset_file}! Please try a valid word."})
                self.wfile.write(err.encode('utf-8'))
                return
                
            try:
                noise_pct = float(req.get('noise_pct', 10))
                noise_pct = max(0.0, min(49.0, noise_pct))  # clamp 0-49
                p_correct = (100.0 - noise_pct) / 100.0
                p_wrong   = (noise_pct / 2.0) / 100.0
                # 100 turns natively implemented!
                history = get_game_history(target, dictionary, p_correct=p_correct, p_wrong=p_wrong, max_turns=100)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"history": history}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Mute standard HTTP logs to keep terminal clean
        pass

if __name__ == "__main__":
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, VisualizerHandler)
    url = f"http://127.0.0.1:{port}"
    print("==========================================================================")
    print(" 🚀 NOISY WORDLE VISUALIZER SERVER ONLINE")
    print(f" -> Local server hosted at: {url}")
    print(" -> Press Ctrl+C in terminal to shutdown.")
    print("==========================================================================")
    
    try:
        webbrowser.open(url)
    except:
        pass
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
