<div align="center">

```
██████╗  █████╗ ████████╗██╗  ██╗███████╗██╗███╗   ██╗██████╗ ███████╗██████╗
██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔══██╗
██████╔╝███████║   ██║   ███████║█████╗  ██║██╔██╗ ██║██║  ██║█████╗  ██████╔╝
██╔═══╝ ██╔══██║   ██║   ██╔══██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
██║     ██║  ██║   ██║   ██║  ██║██║     ██║██║ ╚████║██████╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

### Dynamic Pathfinding Agent
*Real-time A\* & GBFS navigation with live obstacle spawning and automatic replanning*

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-FFD43B?style=flat-square&logo=python&logoColor=black)](https://python.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-11557C?style=flat-square&logo=plotly&logoColor=white)](https://matplotlib.org)
[![Algorithm](https://img.shields.io/badge/A%2A%20%26%20GBFS-Informed%20Search-00ff88?style=flat-square)](#algorithms)
[![Lines](https://img.shields.io/badge/Lines%20of%20Code-~256-orange?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

<br>

> *"The shortest path between two points is only optimal until an obstacle appears."*

</div>

---

<div align="center">

## `[ WHAT IS THIS? ]`

</div>

A **single-file Python visualizer** that brings two classic AI search algorithms to life on an interactive grid. Watch the agent explore, find a path, then adapt on the fly when new walls appear mid-journey — no restarts, just pure real-time replanning.

Built entirely with **Matplotlib** — no game engine, no web server, no external GUI library.

---

<div align="center">

## `[ QUICK START ]`

</div>

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/dynamic-pathfinding-agent.git
cd dynamic-pathfinding-agent

# 2. Install (only one dependency)
pip install matplotlib numpy

# 3. Run
python pathfinding_agent.py
```

> ✅ Works on Windows, macOS, and Linux. Python 3.8+ required.

---

<div align="center">

## `[ FEATURES ]`

</div>

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│   🧠  A* Search          →  Optimal. Uses g(n) + h(n)           │
│   ⚡  Greedy BFS          →  Fast. Uses h(n) only                │
│   📐  3 Heuristics        →  Manhattan · Euclidean · Chebyshev   │
│   🎲  Dynamic Obstacles   →  Random walls spawn mid-flight        │
│   🔄  Auto Replanning     →  Agent reroutes from current cell     │
│   🖊️  Interactive Editor  →  Click to draw/erase walls           │
│   📍  Movable S & G       →  Drag start and goal anywhere        │
│   📊  Live Metrics        →  Nodes visited · Cost · Time (ms)    │
│   🎨  Color Animation     →  Frontier → Visited → Path → Agent   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

<div align="center">

## `[ HOW TO USE ]`

</div>

| Step | Action |
|:----:|--------|
| `1` | Click **Generate** to build a random maze |
| `2` | Choose **Algorithm** — `A*` for optimal, `GBFS` for speed |
| `3` | Choose **Heuristic** — `Manhattan` is the safe default |
| `4` | *(Optional)* Check **Dynamic Obstacles** to enable live walls |
| `5` | Click **▶ Search** and watch the agent navigate |
| `6` | Click any cell in **Wall / Start / Goal** mode to edit the map |

---

<div align="center">

## `[ ALGORITHMS ]`

</div>

### A\* Search
```
f(n) = g(n) + h(n)
         ↑       ↑
    exact cost   heuristic estimate
    from start   to goal
```
Explores nodes in order of *total estimated cost*. **Guaranteed to find the shortest path** as long as the heuristic never overestimates.

---

### Greedy Best-First Search (GBFS)
```
f(n) = h(n)
         ↑
    heuristic only — ignores path cost so far
```
Rushes straight toward the goal. **Faster** than A\* but may find a longer path or get trapped in dead ends.

---

### Heuristics

| Heuristic | Formula | Character |
|-----------|---------|-----------|
| **Manhattan** | `\|Δr\| + \|Δc\|` | Precise for grid movement |
| **Euclidean** | `√(Δr² + Δc²)` | Smooth, slightly underestimates |
| **Chebyshev** | `max(\|Δr\|, \|Δc\|)` | Loose — expands more nodes |

---

### Head-to-Head Comparison

| Property | A\* | GBFS |
|----------|:---:|:----:|
| Finds shortest path | ✅ | ❌ |
| Always finds *a* path | ✅ | ✅ |
| Speed | Moderate | Fast |
| Nodes expanded | More | Fewer |
| Memory usage | Higher | Lower |

---

<div align="center">

## `[ DYNAMIC REPLANNING LOGIC ]`

</div>

```
┌──────────────────────────────────────────────────────────┐
│                                                            │
│   Agent moves one step along path                         │
│           │                                               │
│           ▼                                               │
│   15% chance: random wall spawns somewhere on grid        │
│           │                                               │
│           ▼                                               │
│   Is the new wall on the REMAINING path?                  │
│      │                        │                           │
│     YES                       NO                          │
│      │                        │                           │
│      ▼                        ▼                           │
│   Re-run search()          Continue moving                │
│   from current cell                                       │
│      │                                                    │
│      ├─── Path found? → Update path, keep going           │
│      └─── No path?    → "Blocked!" — stop                 │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

**Key efficiency detail:** The agent only replans when the obstacle is *actually on its current route*. If the wall spawns somewhere irrelevant, the agent ignores it and keeps moving — no wasted computation.

---

<div align="center">

## `[ COLOR LEGEND ]`

</div>

```
  ████  START     — Green   →  Where the agent begins
  ████  GOAL      — Orange  →  The destination
  ████  WALL      — Grey    →  Impassable obstacles
  ████  VISITED   — Blue    →  Nodes the algorithm expanded
  ████  PATH      — Teal    →  The final calculated route
  ████  AGENT     — Purple  →  Live agent in dynamic mode
```

---

<div align="center">

## `[ PROJECT STRUCTURE ]`

</div>

```
dynamic-pathfinding-agent/
│
├── pathfinding_agent.py   ← Entire project. One file. ~256 lines.
├── requirements.txt       ← matplotlib, numpy
└── README.md              ← You are here
```

**`requirements.txt`**
```
matplotlib>=3.5.0
numpy>=1.21.0
```

---

<div align="center">

## `[ BUILT WITH ]`

</div>

| Library | Purpose |
|---------|---------|
| `matplotlib` | GUI window, grid rendering, widgets, animation timer |
| `numpy` | RGB pixel array manipulation for fast cell painting |
| `heapq` | Priority queue for A\* and GBFS open list |
| `random` | Dynamic obstacle spawning |
| `math` | Euclidean heuristic calculation |
| `time` | Execution time measurement |

---

<div align="center">

## `[ LICENSE ]`

MIT — free to use, modify, and distribute.

<br>

---

*Built for AI coursework @ NUCES · Chiniot-Faisalabad Campus*

<br>

**If this helped you, drop a ⭐ — it means a lot.**

</div>
