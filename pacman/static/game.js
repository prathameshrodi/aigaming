const socket = io();
const TILE_SIZE = 30;

// Declare UI elements (initialize them in DOMContentLoaded)
let canvas, ctx, scoreEl, statEp, statEps, statScore, statusEl, logsEl;

function log(msg) {
    console.log(msg); // Always log to console too
    if (logsEl) {
        const div = document.createElement('div');
        div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
        logsEl.prepend(div);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Loaded. Initializing elements...");
    
    canvas = document.getElementById('gameCanvas');
    if(canvas) ctx = canvas.getContext('2d');
    
    scoreEl = document.getElementById('score');
    statEp = document.getElementById('stat-ep');
    statEps = document.getElementById('stat-eps');
    statScore = document.getElementById('stat-score');
    statusEl = document.getElementById('status');
    logsEl = document.getElementById('logs');
    
    const btnStart = document.getElementById('btn-train-start');
    const btnPlay = document.getElementById('btn-play-game');
    const btnStop = document.getElementById('btn-train-stop');

    if (btnStart) {
        btnStart.addEventListener('click', () => {
            log("Requesting Training Start...");
            socket.emit('start_training');
        });
    } else {
        console.error("Start button not found in DOM! Content:", document.body.innerHTML.substring(0, 500));
    }

    if (btnPlay) {
        btnPlay.addEventListener('click', () => {
             log("Requesting Game Start...");
             socket.emit('start_game');
        });
    }

    if (btnStop) {
        btnStop.addEventListener('click', () => {
            log("Requesting Stop...");
            socket.emit('stop_training'); // Reused for both
        });
    }
    
    log("UI Initialized");
});

socket.on('connect', () => {
    if (statusEl) statusEl.innerText = "Status: Connected";
    log("Connected to server WebSocket");
});

socket.on('disconnect', () => {
    if (statusEl) {
        statusEl.innerText = "Status: Disconnected";
        statusEl.style.color = "red";
    }
    log("Disconnected from server");
});

socket.on('connect_error', (err) => {
    log("Connection Error: " + err);
});

socket.on('game_state', (state) => {
    // Debug log to confirm receipt (throttled)
    if (Math.random() < 0.05) log("DEBUG: Rx Frame"); 
    render(state);
    if (scoreEl) {
        if (state.total_pellets) {
             scoreEl.innerText = `${state.score} / ${state.total_pellets}`;
        } else {
             scoreEl.innerText = state.score;
        }
    }
});

socket.on('training_log', (data) => {
    if (statEp) statEp.innerText = data.episode;
    if (statEps) statEps.innerText = data.epsilon.toFixed(3);
    if (statScore) statScore.innerText = data.score.toFixed(1);
});

socket.on('server_log', (data) => {
    log("[TRAIN] " + data.msg);
});

function render(state) {
    if (!ctx) return;
    
    // Clear
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Walls
    if (state.walls) {
        ctx.fillStyle = '#1e3a8a'; // Blue wall color
        for (let y = 0; y < state.walls.length; y++) {
            for (let x = 0; x < state.walls[y].length; x++) {
                if (state.walls[y][x] === 1) {
                    ctx.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
                }
            }
        }
    }

    // Draw Pellets
    ctx.fillStyle = '#ffb8ae'; 
    state.pellets.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x * TILE_SIZE + TILE_SIZE/2, p.y * TILE_SIZE + TILE_SIZE/2, 4, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw Ghosts
    state.ghosts.forEach(g => {
        ctx.fillStyle = `rgb(${g.color[0]}, ${g.color[1]}, ${g.color[2]})`;
        ctx.beginPath();
        ctx.arc(g.x * TILE_SIZE + TILE_SIZE/2, g.y * TILE_SIZE + TILE_SIZE/2, TILE_SIZE/2 - 2, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw Pacman
    ctx.fillStyle = 'yellow';
    ctx.beginPath();
    const p = state.player;
    ctx.arc(p.x * TILE_SIZE + TILE_SIZE/2, p.y * TILE_SIZE + TILE_SIZE/2, TILE_SIZE/2 - 2, 0.2 * Math.PI, 1.8 * Math.PI); 
    ctx.lineTo(p.x * TILE_SIZE + TILE_SIZE/2, p.y * TILE_SIZE + TILE_SIZE/2);
    ctx.fill();
}
