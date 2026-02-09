import os
import csv
import math
import heapq
import tkinter as tk
from tkinter import ttk, messagebox


# -----------------------------
# Map loading / parsing
# -----------------------------
WALL_TOKENS = {"#", "X", "x", "W", "w"}
START_TOKENS = {"S", "s"}
GOAL_TOKENS = {"G", "g", "E", "e"}  # allow E as goal too


def list_csv_files_in_cwd():
    files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    files.sort()
    return files


def parse_cell_token(token: str):
    """
    Return one of: "start", "goal", "wall", "free"
    """
    t = (token or "").strip()

    if t in START_TOKENS:
        return "start"
    if t in GOAL_TOKENS:
        return "goal"
    if t in WALL_TOKENS:
        return "wall"

    # numeric compatibility: 1=wall, 0=free
    # also treat other non-empty non-numeric as free unless it matches above
    if t != "":
        try:
            v = int(float(t))
            if v == 1:
                return "wall"
            else:
                return "free"
        except Exception:
            return "free"

    return "free"


def load_grid_from_csv(path: str):
    """
    Reads CSV into a rectangular grid. Returns:
      grid: list[list[str]]  ("free" or "wall")
      start: (r,c)
      goal: (r,c)
    """
    raw = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Keep row length; allow empty cells
            raw.append(row)

    if not raw:
        raise ValueError("CSV is empty.")

    # Normalize to rectangular
    cols = max(len(r) for r in raw)
    for r in raw:
        if len(r) < cols:
            r.extend([""] * (cols - len(r)))

    start = None
    goal = None
    grid = [["free" for _ in range(cols)] for _ in range(len(raw))]

    for i in range(len(raw)):
        for j in range(cols):
            kind = parse_cell_token(raw[i][j])
            if kind == "start":
                if start is not None:
                    raise ValueError("Multiple start cells found. Please keep exactly one start.")
                start = (i, j)
                grid[i][j] = "free"
            elif kind == "goal":
                if goal is not None:
                    raise ValueError("Multiple goal cells found. Please keep exactly one goal.")
                goal = (i, j)
                grid[i][j] = "free"
            elif kind == "wall":
                grid[i][j] = "wall"
            else:
                grid[i][j] = "free"

    if start is None or goal is None:
        raise ValueError("Start (S) or Goal (G) not found in the CSV.")

    return grid, start, goal


# -----------------------------
# A* with step snapshots
# -----------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors_4(grid, node):
    R, C = len(grid), len(grid[0])
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != "wall":
            yield (nr, nc)


def reconstruct_path(came_from, start, goal):
    if goal not in came_from and goal != start:
        return []
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path


def open_list_sorted_display(open_heap, g_score, goal):
    """
    Produce a sorted, de-duplicated view of open list:
    show only entries consistent with current g_score (ignore stale heap entries).
    Return list of tuples: (node, f, g, h)
    """
    seen = set()
    items = []
    for f, tie, node in open_heap:
        if node in seen:
            continue
        if node not in g_score:
            continue
        g = g_score[node]
        h = manhattan(node, goal)
        f_true = g + h
        # Only keep entries where heap f matches current best f (avoid stale)
        # But because we push with f_true, a stale entry will differ.
        if f != f_true:
            continue
        seen.add(node)
        items.append((node, f_true, g, h))

    items.sort(key=lambda x: (x[1], x[2], x[0][0], x[0][1]))
    return items


def astar_generate_snapshots(grid, start, goal):
    """
    Runs A* once and records snapshots after each iteration.
    Snapshot includes:
      - current node (or None)
      - closed set
      - open list display items
      - came_from mapping
      - g_score mapping
      - status string
      - path (current best path to current or final path to goal)
    """
    open_heap = []
    tie = 0

    came_from = {}
    g_score = {start: 0}
    h0 = manhattan(start, goal)
    heapq.heappush(open_heap, (0 + h0, tie, start))
    tie += 1

    closed = set()
    snapshots = []

    def push_snapshot(current, status):
        open_items = open_list_sorted_display(open_heap, g_score, goal)
        path = []
        if status == "FOUND":
            path = reconstruct_path(came_from, start, goal)
        else:
            if current is not None and current in g_score:
                # Show best-known path to current as "progress"
                path = reconstruct_path(came_from, start, current)

        snapshots.append({
            "current": current,
            "closed": set(closed),
            "open_items": list(open_items),  # list of (node, f, g, h)
            "came_from": dict(came_from),
            "g_score": dict(g_score),
            "status": status,
            "path": list(path),
        })

    # Initial snapshot before any pop (queue contains start)
    push_snapshot(current=None, status="INIT")

    found = False

    while open_heap:
        # pop the best valid entry
        f, _, current = heapq.heappop(open_heap)
        if current not in g_score:
            continue
        g = g_score[current]
        h = manhattan(current, goal)
        if f != g + h:
            # stale
            continue

        # Record snapshot: node selected for expansion (before expanding neighbors)
        push_snapshot(current=current, status="POP")

        if current == goal:
            found = True
            push_snapshot(current=current, status="FOUND")
            break

        closed.add(current)

        # Expand neighbors
        for nb in neighbors_4(grid, current):
            if nb in closed:
                continue
            tentative_g = g + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_nb = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f_nb, tie, nb))
                tie += 1

        # Snapshot after expanding neighbors (queue updated)
        push_snapshot(current=current, status="EXPAND_DONE")

    if not found:
        push_snapshot(current=None, status="NO_PATH")

    return snapshots


# -----------------------------
# Tkinter UI
# -----------------------------
class AStarVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A* Visualizer (CSV maps in current folder)")
        self.geometry("1100x700")

        self.grid_data = None
        self.start = None
        self.goal = None
        self.snapshots = []
        self.step_idx = 0

        self.cell_size = 28
        self.pad = 12

        self._build_ui()
        self._refresh_csv_dropdown()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text="Map (CSV):").pack(side=tk.LEFT)
        self.csv_var = tk.StringVar(value="")
        self.csv_dropdown = ttk.Combobox(top, textvariable=self.csv_var, state="readonly", width=40)
        self.csv_dropdown.pack(side=tk.LEFT, padx=8)
        self.csv_dropdown.bind("<<ComboboxSelected>>", lambda e: self.load_selected_map())

        ttk.Button(top, text="Reload CSV list", command=self._refresh_csv_dropdown).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load map", command=self.load_selected_map).pack(side=tk.LEFT, padx=6)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left: Canvas
        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: Open list + status
        right = ttk.Frame(mid, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        ttk.Label(right, text="Priority Queue (Open List) items:").pack(anchor="w")
        ttk.Label(right, text='Format: f(g+h)   (sorted by f)').pack(anchor="w", pady=(0, 6))

        self.open_list = tk.Listbox(right, height=22)
        self.open_list.pack(fill=tk.BOTH, expand=False)

        self.status_label = ttk.Label(right, text="Status: -")
        self.status_label.pack(anchor="w", pady=(10, 2))

        self.step_label = ttk.Label(right, text="Step: 0 / 0")
        self.step_label.pack(anchor="w")

        self.current_label = ttk.Label(right, text="Current: -")
        self.current_label.pack(anchor="w", pady=(2, 10))

        # --- tooltip for hover coordinate (follows mouse) ---
        self.tooltip = tk.Toplevel(self)
        self.tooltip.withdraw()
        self.tooltip.overrideredirect(True)  # no window decorations
        self.tooltip.attributes("-topmost", True)

        self.tooltip_label = tk.Label(
            self.tooltip,
            text="",
            bg="#ffffe0",
            fg="black",
            bd=1,
            relief="solid",
            padx=6,
            pady=3
        )
        self.tooltip_label.pack()

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)


        # Bottom controls
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.prev_btn = ttk.Button(bottom, text="Prev step", command=self.prev_step)
        self.prev_btn.pack(side=tk.LEFT)

        self.next_btn = ttk.Button(bottom, text="Next step", command=self.next_step)
        self.next_btn.pack(side=tk.LEFT, padx=8)

        ttk.Button(bottom, text="Reset to start", command=self.reset_steps).pack(side=tk.LEFT, padx=8)

        ttk.Label(bottom, text="Cell size:").pack(side=tk.LEFT, padx=(30, 6))
        self.size_var = tk.IntVar(value=self.cell_size)
        size_spin = ttk.Spinbox(bottom, from_=12, to=60, textvariable=self.size_var, width=6, command=self._apply_cell_size)
        size_spin.pack(side=tk.LEFT)

    def _apply_cell_size(self):
        try:
            v = int(self.size_var.get())
            self.cell_size = max(10, min(80, v))
            self.render()
        except Exception:
            pass

    def _refresh_csv_dropdown(self):
        files = list_csv_files_in_cwd()
        self.csv_dropdown["values"] = files
        if files and self.csv_var.get() not in files:
            self.csv_var.set(files[0] if files else "")

    def load_selected_map(self):
        path = self.csv_var.get().strip()
        if not path:
            messagebox.showinfo("No CSV", "No CSV selected/found in current folder.")
            return
        try:
            grid, start, goal = load_grid_from_csv(path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.grid_data = grid
        self.start = start
        self.goal = goal

        # Precompute snapshots so Prev/Next is easy and exact.
        self.snapshots = astar_generate_snapshots(grid, start, goal)
        self.step_idx = 0
        self.render()
    
    def on_mouse_leave(self, event):
        # Hide tooltip when mouse leaves the canvas
        if hasattr(self, "tooltip"):
            self.tooltip.withdraw()

    def on_mouse_move(self, event):
        if not self.grid_data:
            self.tooltip.withdraw()
            return

        # Convert pixel -> grid coords
        x = event.x - self.pad
        y = event.y - self.pad

        if x < 0 or y < 0:
            self.tooltip.withdraw()
            return

        col = x // self.cell_size
        row = y // self.cell_size

        R, C = len(self.grid_data), len(self.grid_data[0])

        if not (0 <= row < R and 0 <= col < C):
            self.tooltip.withdraw()
            return

        # Update tooltip text
        snap = self.snapshots[self.step_idx] if self.snapshots else None
        g_score = snap["g_score"] if snap else {}
        node = (row, col)

        if node in g_score:
            g = g_score[node]
            h = manhattan(node, self.goal)
            f = g + h
            self.tooltip_label.config(text=f"({row},{col})  f={f} ({g}+{h})")
        else:
            self.tooltip_label.config(text=f"({row}, {col})")


        # Place tooltip near the mouse (in screen coordinates)
        offset_x, offset_y = 14, 14
        screen_x = self.canvas.winfo_rootx() + event.x + offset_x
        screen_y = self.canvas.winfo_rooty() + event.y + offset_y
        self.tooltip.geometry(f"+{screen_x}+{screen_y}")
        self.tooltip.deiconify()


    def reset_steps(self):
        if not self.snapshots:
            return
        self.step_idx = 0
        self.render()

    def prev_step(self):
        if not self.snapshots:
            return
        if self.step_idx > 0:
            self.step_idx -= 1
        self.render()

    def next_step(self):
        if not self.snapshots:
            return
        if self.step_idx < len(self.snapshots) - 1:
            self.step_idx += 1
        self.render()

    def render(self):
        self.canvas.delete("all")
        self.open_list.delete(0, tk.END)

        if not self.grid_data or not self.snapshots:
            self.status_label.config(text="Status: -")
            self.step_label.config(text="Step: 0 / 0")
            self.current_label.config(text="Current: -")
            return

        snap = self.snapshots[self.step_idx]
        g_score = snap["g_score"] if snap else {}
        grid = self.grid_data
        R, C = len(grid), len(grid[0])
        cs = self.cell_size

        # Resize canvas scroll region
        w = self.pad * 2 + C * cs
        h = self.pad * 2 + R * cs
        self.canvas.config(scrollregion=(0, 0, w, h))

        closed = snap["closed"]
        current = snap["current"]
        open_items = snap["open_items"]
        path = set(snap["path"])

        # Draw cells
        for r in range(R):
            for c in range(C):
                x0 = self.pad + c * cs
                y0 = self.pad + r * cs
                x1 = x0 + cs
                y1 = y0 + cs

                node = (r, c)
                fill = "white"

                if grid[r][c] == "wall":
                    fill = "black"
                else:
                    if node in closed:
                        fill = "#e8e8e8"  # closed
                    if node in path:
                        fill = "#d8c8ff"  # path
                    if any(node == it[0] for it in open_items):
                        fill = "#cfefff"  # open
                    if node == current:
                        fill = "#ffeaa7"  # current

                # start/goal override
                if node == self.start:
                    fill = "#b7f7b7"
                if node == self.goal:
                    fill = "#ffb3b3"
                if node == current:
                    # current highlight on top (unless wall)
                    if grid[r][c] != "wall":
                        fill = "#ffeaa7"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#cccccc")
                # Optional: draw coordinates or small dot; keep clean
                if node == self.start:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text="S")
                elif node == self.goal:
                    self.canvas.create_text((x0+x1)//2, (y0+y1)//2, text="G")
                
                # --- draw f value for discovered cells (in g_score) ---
                node = (r, c)
                if grid[r][c] != "wall" and node in g_score:
                    # Optional: avoid clutter if cells are too small
                    if cs >= 22:
                        g = g_score[node]
                        h = manhattan(node, self.goal)
                        f = g + h

                        # Don't overwrite S/G labels (optional)
                        if node != self.start and node != self.goal:
                            self.canvas.create_text(
                                (x0 + x1) // 2,
                                (y0 + y1) // 2,
                                text=str(f),
                                font=("TkDefaultFont", max(8, cs // 3))
                            )

        # Populate open list box (priority queue display)
        for node, f, g, h in open_items:
            self.open_list.insert(tk.END, f"{f}({g}+{h})  @ {node}")

        status = snap["status"]
        self.status_label.config(text=f"Status: {status}")
        self.step_label.config(text=f"Step: {self.step_idx + 1} / {len(self.snapshots)}")
        self.current_label.config(text=f"Current: {current if current is not None else '-'}")

        # Disable/enable buttons
        self.prev_btn.config(state=("disabled" if self.step_idx == 0 else "normal"))
        self.next_btn.config(state=("disabled" if self.step_idx == len(self.snapshots) - 1 else "normal"))


if __name__ == "__main__":
    app = AStarVisualizer()
    app.mainloop()
