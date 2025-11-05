"""
robomouse_animate.py
Generate a random maze, solve it with A*, animate the robot following the path,
save animation as GIF (and optionally MP4).

Run:
    python robomouse_animate.py

Requirements:
    pip install numpy matplotlib pillow
"""
import os
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -------------------------
# Maze generation (recursive backtracker)
# -------------------------
def generate_maze(width=41, height=41, seed=12345):
    assert width % 2 == 1 and height % 2 == 1, "Use odd dimensions (e.g., 41x41)"
    rng = random.Random(seed)
    grid = np.ones((height, width), dtype=np.uint8)
    stack = [(1, 1)]
    grid[1,1] = 0
    dirs = [(2,0),(-2,0),(0,2),(0,-2)]
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx,dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width-1 and 1 <= ny < height-1 and grid[ny, nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            nx, ny = rng.choice(neighbors)
            wx, wy = (x + nx)//2, (y + ny)//2
            grid[wy, wx] = 0
            grid[ny, nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()
    return grid

# -------------------------
# A* pathfinding (4-connected)
# -------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors4(node, grid):
    x,y = node
    h,w = grid.shape
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx,ny = x+dx, y+dy
        if 0 <= nx < w and 0 <= ny < h and grid[ny,nx] == 0:
            yield (nx, ny)

def a_star(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start,goal), 0, start, None))
    came_from = {}
    cost_so_far = {start: 0}
    while open_heap:
        _, cost, current, parent = heapq.heappop(open_heap)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            # reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path
        for nbr in neighbors4(current, grid):
            new_cost = cost_so_far[current] + 1
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                priority = new_cost + heuristic(nbr, goal)
                heapq.heappush(open_heap, (priority, new_cost, nbr, current))
    return None

# -------------------------
# Animation & saving
# -------------------------
def animate_and_save(width=41, height=41, seed=12345, out_dir='robomouse_output',
                     gif_name=None, mp4_name=None, fps=12, robot_size=8):
    os.makedirs(out_dir, exist_ok=True)
    if gif_name is None:
        gif_name = f'maze_anim_seed_{seed}.gif'
    if mp4_name is None:
        mp4_name = f'maze_anim_seed_{seed}.mp4'
    gif_path = os.path.join(out_dir, gif_name)
    mp4_path = os.path.join(out_dir, mp4_name)

    grid = generate_maze(width, height, seed)
    start = (1,1)
    free_cells = [(x,y) for y in range(grid.shape[0]) for x in range(grid.shape[1]) if grid[y,x] == 0]
    # pick farthest free cell from start as goal
    goal = max(free_cells, key=lambda c: heuristic(c, start))
    # ensure start/goal free
    grid[start[1], start[0]] = 0
    grid[goal[1], goal[0]] = 0

    path = a_star(grid, start, goal)
    if path is None:
        raise RuntimeError("No path found - regenerate or change seed/size.")

    h,w = grid.shape
    # Create an RGB image base (walls black, free white)
    img = np.zeros((h,w,3), dtype=float)
    img[grid==1] = [0,0,0]   # walls
    img[grid==0] = [1,1,1]   # free
    # Create a faint path layer for context
    path_img = img.copy()
    for (x,y) in path:
        path_img[y,x] = [0.7, 0.9, 0.7]

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_axis_off()
    ax.imshow(path_img, origin='lower', interpolation='nearest')

    # Markers
    sx,sy = start; gx,gy = goal
    ax.scatter([sx],[sy], s=60, marker='s')  # start
    ax.scatter([gx],[gy], s=60, marker='s')  # goal
    robot_marker = ax.scatter([sx],[sy], s=robot_size**2, marker='o', edgecolors='k')

    def init():
        robot_marker.set_offsets([[sx, sy]])
        return (robot_marker,)

    def update(frame):
        x,y = path[frame]
        robot_marker.set_offsets([[x, y]])
        return (robot_marker,)

    frames = len(path)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames,
                                   interval=1000/fps, blit=True)

    # Save GIF (requires pillow)
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        print(f"Saved GIF: {gif_path}")
    except Exception as e:
        print("GIF save failed (pillow installed?):", e)

    # Optionally save MP4 (requires ffmpeg available on PATH)
    try:
        anim.save(mp4_path, writer='ffmpeg', fps=fps)
        print(f"Saved MP4: {mp4_path}")
    except Exception as e:
        # Not fatal — ffmpeg may not be installed
        print("MP4 save failed (ffmpeg available?):", e)

    plt.close(fig)
    return gif_path, mp4_path, grid, start, goal, path

# -------------------------
# Run script
# -------------------------
if __name__ == '__main__':
    # Customize these parameters as you like:
    maze_w = 41        # odd number
    maze_h = 41        # odd number
    seed = 12345       # change to get different maze
    outdir = 'robomouse_output'
    gif_path, mp4_path, grid, start, goal, path = animate_and_save(width=maze_w, height=maze_h,
                                                                  seed=seed, out_dir=outdir,
                                                                  fps=12, robot_size=8)
    print("Done. Files written (if saves succeeded):")
    print("  GIF:", gif_path)
    print("  MP4:", mp4_path)
    print("Path length:", len(path))
