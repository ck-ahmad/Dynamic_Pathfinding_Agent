"""
Dynamic Pathfinding Agent — A* & GBFS with Matplotlib
Run: python pathfinding_agent.py
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import numpy as np
import heapq, random, math, time

# ── Constants ──────────────────────────────────────────────
ROWS, COLS = 20, 25
EMPTY, WALL, START, GOAL, VISITED, PATH, AGENT = 0,1,2,3,4,5,6

COLORS = {
    EMPTY:   [0.10, 0.10, 0.15],
    WALL:    [0.30, 0.30, 0.38],
    START:   [0.30, 0.85, 0.45],
    GOAL:    [0.98, 0.70, 0.30],
    VISITED: [0.35, 0.55, 0.95],
    PATH:    [0.20, 0.90, 0.60],
    AGENT:   [0.85, 0.55, 0.95],
}

# ── Heuristic ──────────────────────────────────────────────
def h(a, b, kind):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return (dr+dc) if kind=="Manhattan" else math.sqrt(dr*dr+dc*dc) if kind=="Euclidean" else max(dr,dc)

# ── Search (A* / GBFS) ─────────────────────────────────────
def search(grid, start, goal, algo, hkind):
    rows, cols = len(grid), len(grid[0])
    g = {start: 0};  came = {};  heap = [];  seen = set();  closed = set()
    heapq.heappush(heap, (h(start,goal,hkind), start[0], start[1]))
    visited = []
    while heap:
        _, r, c = heapq.heappop(heap)
        if (r,c) in closed: continue
        closed.add((r,c));  visited.append((r,c))
        if (r,c) == goal:
            path = []
            cur = goal
            while cur in came: path.append(cur); cur = came[cur]
            return list(reversed([start]+path)), visited, round((time.time())*1000%1e6, 1)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            nb = (nr,nc)
            if not(0<=nr<rows and 0<=nc<cols) or grid[nr][nc]==WALL or nb in closed: continue
            ng = g[(r,c)] + 1
            if algo=="A*" and nb in g and g[nb]<=ng: continue
            g[nb] = ng;  came[nb] = (r,c)
            f = ng + h(nb,goal,hkind) if algo=="A*" else h(nb,goal,hkind)
            heapq.heappush(heap, (f, nr, nc));  seen.add(nb)
    return [], visited, 0

# ── App ────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.start = (1,1);  self.goal = (ROWS-2, COLS-2)
        self.algo = "A*";  self.heur = "Manhattan";  self.dyn = False
        self.running = False;  self.timer = None
        self.agent_path = [];  self.agent_idx = 0
        self._setup_fig()
        self._new_grid(None)
        plt.show()

    def _setup_fig(self):
        self.fig = plt.figure(figsize=(14,7), facecolor="#111118")
        self.fig.canvas.manager.set_window_title("Pathfinding Agent")
        self.ax = self.fig.add_axes([0.01, 0.02, 0.60, 0.95])
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.img = None

        tc = "#ccccdd";  fc = "#1a1a26"
        self.fig.text(0.63, 0.97, "Dynamic Pathfinding Agent", color="#66ffaa",
                      fontsize=11, fontweight="bold", fontfamily="monospace", va="top")

        # Algorithm
        self.fig.text(0.63, 0.90, "Algorithm", color=tc, fontsize=8, fontfamily="monospace")
        ax1 = self.fig.add_axes([0.63, 0.81, 0.13, 0.08], facecolor=fc)
        self.r_algo = RadioButtons(ax1, ["A*","GBFS"], activecolor="#66ffaa")
        [l.set(color=tc, fontsize=9) for l in self.r_algo.labels]
        self.r_algo.on_clicked(lambda v: setattr(self,"algo",v))

        # Heuristic
        self.fig.text(0.63, 0.80, "Heuristic", color=tc, fontsize=8, fontfamily="monospace")
        ax2 = self.fig.add_axes([0.63, 0.68, 0.16, 0.11], facecolor=fc)
        self.r_heur = RadioButtons(ax2, ["Manhattan","Euclidean","Chebyshev"], activecolor="#66ffaa")
        [l.set(color=tc, fontsize=9) for l in self.r_heur.labels]
        self.r_heur.on_clicked(lambda v: setattr(self,"heur",v))

        # Edit mode
        self.fig.text(0.63, 0.67, "Edit Mode (click grid)", color=tc, fontsize=8, fontfamily="monospace")
        ax3 = self.fig.add_axes([0.63, 0.56, 0.19, 0.10], facecolor=fc)
        self.r_edit = RadioButtons(ax3, ["Wall","Start","Goal"], activecolor="#ffdd66")
        [l.set(color=tc, fontsize=9) for l in self.r_edit.labels]

        # Dynamic checkbox
        ax4 = self.fig.add_axes([0.63, 0.49, 0.25, 0.05], facecolor=fc)
        self.chk = CheckButtons(ax4, ["Dynamic Obstacles"], [False])
        [l.set(color=tc, fontsize=8) for l in self.chk.labels]
        self.chk.on_clicked(lambda _: setattr(self,"dyn", not self.dyn))

        # Buttons
        def btn(rect, label, fg, bg):
            a = self.fig.add_axes(rect)
            b = Button(a, label, color=bg, hovercolor="#444466")
            b.label.set(color=fg, fontsize=9, fontfamily="monospace")
            return b

        self.b_gen   = btn([0.63,0.41,0.13,0.05], "Generate",  "#ccccdd","#2a2a3a")
        self.b_start = btn([0.63,0.34,0.13,0.05], "▶ Search",  "#66ffaa","#1a3a26")
        self.b_stop  = btn([0.63,0.27,0.13,0.05], "■ Stop",    "#ff8888","#3a1a1a")
        self.b_clr   = btn([0.63,0.20,0.13,0.05], "↺ Clear",   "#88aaff","#1a1a3a")
        self.b_gen.on_clicked(self._new_grid);  self.b_start.on_clicked(self._run)
        self.b_stop.on_clicked(self._stop);     self.b_clr.on_clicked(self._clear)

        # Metrics
        self.fig.text(0.80, 0.90, "── Metrics ──", color=tc, fontsize=8, fontfamily="monospace")
        self.t_vis  = self.fig.text(0.80, 0.85, "Visited : 0",  color=tc, fontsize=8, fontfamily="monospace")
        self.t_cost = self.fig.text(0.80, 0.80, "Cost    : 0",  color=tc, fontsize=8, fontfamily="monospace")
        self.t_time = self.fig.text(0.80, 0.75, "Time ms : 0",  color=tc, fontsize=8, fontfamily="monospace")
        self.t_stat = self.fig.text(0.80, 0.69, "Status  : Ready", color="#ffdd66", fontsize=9,
                                    fontweight="bold", fontfamily="monospace")

        # Legend
        self.fig.text(0.80, 0.63, "── Legend ──", color=tc, fontsize=8, fontfamily="monospace")
        for i,(ct,lb) in enumerate([(START,"Start"),(GOAL,"Goal"),(WALL,"Wall"),
                                     (VISITED,"Visited"),(PATH,"Path"),(AGENT,"Agent")]):
            y = 0.58 - i*0.05
            self.fig.add_artist(plt.Rectangle((0.80,y),0.03,0.03, color=COLORS[ct],
                                               transform=self.fig.transFigure, clip_on=False))
            self.fig.text(0.84, y+0.005, lb, color="#aaaacc", fontsize=8, fontfamily="monospace")

        self.fig.canvas.mpl_connect("button_press_event", self._click)

    def _to_img(self):
        g = [r[:] for r in self.grid]
        g[self.start[0]][self.start[1]] = START
        g[self.goal[0]][self.goal[1]]   = GOAL
        img = np.zeros((ROWS, COLS, 3))
        for r in range(ROWS):
            for c in range(COLS):
                img[r,c] = COLORS[g[r][c]]
        return img

    def _draw(self):
        img = self._to_img()
        if self.img is None:
            self.img = self.ax.imshow(img, interpolation="nearest", aspect="equal",
                                      extent=[-0.5,COLS-0.5,ROWS-0.5,-0.5])
            self.ax.text(self.start[1],self.start[0],"S",ha="center",va="center",
                         color="#111",fontsize=8,fontweight="bold")
            self.ax.text(self.goal[1], self.goal[0], "G",ha="center",va="center",
                         color="#111",fontsize=8,fontweight="bold")
        else:
            self.img.set_data(img)
        self.fig.canvas.draw_idle()

    def _paint(self, cells, ctype):
        arr = np.array(self.img.get_array())
        for r,c in cells:
            if (r,c) not in (self.start, self.goal): arr[r,c] = COLORS[ctype]
        self.img.set_data(arr);  self.fig.canvas.draw_idle()

    def _new_grid(self, _):
        self._stop(None)
        self.grid = [[WALL if (r==0 or r==ROWS-1 or c==0 or c==COLS-1)
                      else (WALL if random.random()<0.25 else EMPTY)
                      for c in range(COLS)] for r in range(ROWS)]
        self.grid[self.start[0]][self.start[1]] = EMPTY
        self.grid[self.goal[0]][self.goal[1]]   = EMPTY
        self._draw();  self._status("Ready", 0,0,0)

    def _click(self, e):
        if self.running or e.inaxes != self.ax: return
        c,r = int(round(e.xdata)), int(round(e.ydata))
        if not(0<=r<ROWS and 0<=c<COLS): return
        m = self.r_edit.value_selected
        if m=="Wall" and (r,c) not in (self.start,self.goal):
            self.grid[r][c] = EMPTY if self.grid[r][c]==WALL else WALL;  self._draw()
        elif m=="Start" and self.grid[r][c]!=WALL and (r,c)!=self.goal:
            self.start=(r,c);  self._draw()
        elif m=="Goal"  and self.grid[r][c]!=WALL and (r,c)!=self.start:
            self.goal=(r,c);   self._draw()

    def _run(self, _):
        if self.running: return
        self._clear(None);  self.running = True;  self._status("Searching...",0,0,0)
        t0 = time.time()
        path, vis, _ = search(self.grid, self.start, self.goal, self.algo, self.heur)
        elapsed = round((time.time()-t0)*1000, 2)
        if not path:
            self._status("No path found!",len(vis),0,elapsed);  self.running=False;  return
        self._status("Animating...", len(vis), len(path)-1, elapsed)
        step = [0]
        def tick():
            if not self.running: return
            if step[0] < len(vis):
                self._paint(vis[step[0]:step[0]+6], VISITED);  step[0]+=6
                self._sched(tick, 15)
            elif step[0] < len(vis)+len(path):
                self._paint([path[step[0]-len(vis)]], PATH);  step[0]+=1
                self._sched(tick, 25)
            else:
                self._paint([self.start],START);  self._paint([self.goal],GOAL)
                self._status("Path Found! ✓", len(vis), len(path)-1, elapsed)
                if self.dyn: self.agent_path=path; self.agent_idx=0; self._sched(self._dyn,200)
                else: self.running=False
        tick()

    def _dyn(self):
        if not self.running: return
        if self.agent_idx >= len(self.agent_path)-1:
            self._status("Goal Reached! ✓",0,0,0);  self.running=False;  return
        prev = self.agent_path[self.agent_idx];  self.agent_idx+=1
        cur  = self.agent_path[self.agent_idx]
        self._paint([prev], PATH);  self._paint([cur], AGENT);  self._paint([self.goal], GOAL)
        # Random wall
        if random.random()<0.15:
            r,c = random.randint(1,ROWS-2), random.randint(1,COLS-2)
            if (r,c) not in (cur,self.start,self.goal) and self.grid[r][c]==EMPTY:
                self.grid[r][c]=WALL;  self._paint([(r,c)],WALL)
                if (r,c) in self.agent_path[self.agent_idx:]:
                    p,v,t = search(self.grid,cur,self.goal,self.algo,self.heur)
                    if p: self.agent_path=p; self.agent_idx=0; self._status("Replanned ✓",len(v),len(p)-1,t)
                    else: self._status("Blocked!",0,0,0); self.running=False; return
        self._sched(self._dyn, 130)

    def _sched(self, fn, ms):
        if self.timer:
            try: self.timer.stop()
            except: pass
        self.timer = self.fig.canvas.new_timer(interval=ms)
        self.timer.single_shot=True;  self.timer.add_callback(fn);  self.timer.start()

    def _stop(self, _):
        if self.timer:
            try: self.timer.stop()
            except: pass
        self.running=False

    def _clear(self, _):
        self._stop(None)
        if self.grid: self._draw()
        self._status("Ready",0,0,0)

    def _status(self, msg, vis, cost, t):
        self.t_vis.set_text(f"Visited : {vis}")
        self.t_cost.set_text(f"Cost    : {cost}")
        self.t_time.set_text(f"Time ms : {t}")
        col = "#66ffaa" if "✓" in msg else "#ff8888" if "No path" in msg or "Blocked" in msg else "#ffdd66"
        self.t_stat.set_text(f"Status  : {msg}");  self.t_stat.set_color(col)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    App()