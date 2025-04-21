import pygame
import numpy as np
import math
from collections import deque
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import csv
import heapq
import random  

# ================== PARAMETERS ==================
MAP_SIZE = (800, 800)
GRID_SIZE = 20
GRID_WIDTH = MAP_SIZE[0] // GRID_SIZE
GRID_HEIGHT = MAP_SIZE[1] // GRID_SIZE
OBSTACLE_COUNT = 100
ROBOT_RADIUS = 10
INFLATION_RADIUS = int(ROBOT_RADIUS / GRID_SIZE + 1)
LIDAR_RANGE = 100
LIDAR_ANGLE_RES = 10
LIDAR_NOISE_STD = 2
GOAL_POS = (750, 750)

# ============== HELPER FUNCTIONS ==============
def create_random_map():
    obstacles = []
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.uint8)
    for _ in range(OBSTACLE_COUNT):
        x, y = np.random.randint(0, GRID_WIDTH - 1), np.random.randint(0, GRID_HEIGHT - 1)
        grid[x, y] = 1
        obstacles.append(pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Inflate obstacles
    inflated_grid = grid.copy()
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if grid[x, y] == 1:
                for dx in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                    for dy in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                            inflated_grid[nx, ny] = 1
    return obstacles, inflated_grid

def draw_obstacles(screen, obstacles):
    for obs in obstacles:
        pygame.draw.rect(screen, (0, 0, 0), obs)

def check_collision(x, y, obstacles):
    point = pygame.Rect(x - ROBOT_RADIUS, y - ROBOT_RADIUS, 2 * ROBOT_RADIUS, 2 * ROBOT_RADIUS)
    return any(obs.colliderect(point) for obs in obstacles)

def grid_from_pos(pos):
    return int(pos[0] // GRID_SIZE), int(pos[1] // GRID_SIZE)

def pos_from_grid(cell):
    return cell[0] * GRID_SIZE + GRID_SIZE // 2, cell[1] * GRID_SIZE + GRID_SIZE // 2

def a_star(grid, start, goal):
    start, goal = tuple(start), tuple(goal)
    queue = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]

    while queue:
        _, current = heapq.heappop(queue)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in directions:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and grid[nx, ny] == 0:
                new_cost = cost_so_far[current] + math.hypot(dx, dy)
                neighbor = (nx, ny)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + math.hypot(goal[0]-nx, goal[1]-ny)
                    heapq.heappush(queue, (priority, neighbor))
                    came_from[neighbor] = current

    return []

def simulate_lidar(x, y, theta, obstacles):
    angles = np.arange(-90, 91, LIDAR_ANGLE_RES)
    scans = []
    for angle in angles:
        angle_rad = math.radians(theta + angle)
        for r in range(0, LIDAR_RANGE, 2):
            rx = x + r * math.cos(angle_rad)
            ry = y + r * math.sin(angle_rad)
            if any(obs.collidepoint(rx, ry) for obs in obstacles):
                scans.append([rx + np.random.normal(0, LIDAR_NOISE_STD), ry + np.random.normal(0, LIDAR_NOISE_STD)])
                break
        else:
            scans.append([x + LIDAR_RANGE * math.cos(angle_rad), y + LIDAR_RANGE * math.sin(angle_rad)])
    return np.array(scans)

def draw_lidar(screen, robot_pos, scans):
    for pt in scans:
        pygame.draw.line(screen, (200, 200, 200), robot_pos, pt, 1)

def draw_robot(screen, pos, heading, color=(0, 0, 0)):
    pygame.draw.circle(screen, color, pos, ROBOT_RADIUS)
    front = (pos[0] + 15 * math.cos(math.radians(heading)),
             pos[1] + 15 * math.sin(math.radians(heading)))
    pygame.draw.circle(screen, (255, 0, 0), front, 3)

def draw_goal(screen):
    pygame.draw.polygon(screen, (255, 165, 0), [(GOAL_POS[0], GOAL_POS[1]),
                                                 (GOAL_POS[0]-5, GOAL_POS[1]+10),
                                                 (GOAL_POS[0]+5, GOAL_POS[1]+10)])

def draw_path(screen, path):
    for cell in path:
        pos = pos_from_grid(cell)
        pygame.draw.circle(screen, (0, 255, 255), pos, 3)

def icp(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.eye(2), np.zeros(2), 1e3
    neigh = NearestNeighbors(n_neighbors=1).fit(B)
    distances, indices = neigh.kneighbors(A)
    A_matched = A
    B_matched = B[indices.flatten()]
    centroid_A = np.mean(A_matched, axis=0)
    centroid_B = np.mean(B_matched, axis=0)
    H = ((A_matched - centroid_A).T @ (B_matched - centroid_B))
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    A_transformed = (R @ A_matched.T).T + t
    rmse = np.sqrt(np.mean(np.sum((A_transformed - B_matched)**2, axis=1)))
    return R, t, rmse

def apply_transform(points, R, t):
    return (R @ points.T).T + t

def caer(scan, map_points):
    if len(scan) == 0 or len(map_points) == 0:
        return 1e3
    tree = KDTree(map_points)
    dists, _ = tree.query(scan)
    return np.mean(dists)

def export_logs(log, filename="cbgl_log.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Step", "CAER", "ICP_RMSE"])
        for i, (c, r) in enumerate(log):
            writer.writerow([i, c, r])

def generate_hypotheses(grid, num_hypotheses):
    """Generates hypotheses primarily in free space."""
    free_cells = np.where(grid == 0)
    hypotheses = []
    num_free_cells = len(free_cells[0])
    if num_free_cells == 0:
        for _ in range(num_hypotheses):
            x = random.uniform(0, MAP_SIZE[0])
            y = random.uniform(0, MAP_SIZE[1])
            theta = random.uniform(-math.pi, math.pi)
            hypotheses.append((x, y, theta))
        return hypotheses

    for _ in range(num_hypotheses):
        idx = random.randint(0, num_free_cells - 1)
        grid_x, grid_y = free_cells[0][idx], free_cells[1][idx]
        x = grid_x * GRID_SIZE + GRID_SIZE // 2
        y = grid_y * GRID_SIZE + GRID_SIZE // 2
        theta = random.uniform(-math.pi, math.pi)
        hypotheses.append((x, y, theta))
    return hypotheses

def visualize_hypotheses(screen, hypotheses_caer, top_k, obstacles):
    """Visualizes the top-k pose hypotheses on the Pygame screen."""
    for i in range(min(top_k, len(hypotheses_caer))):
        caer_value, h_x, h_y, h_theta = hypotheses_caer[i]
        color = (0, 255 - int((i / top_k) * 255), int((i / top_k) * 255))
        draw_robot(screen, (h_x, h_y), math.degrees(h_theta), color=color)
        map_scan = simulate_lidar(h_x, h_y, h_theta, obstacles)
        draw_lidar(screen, (h_x, h_y), map_scan)

def cbgl_localize(real_scan, obstacles, grid, num_hypotheses, top_k, screen=None, robot_pos=None, robot_theta=None):
    """Implements the CBGL algorithm to estimate the robot's pose."""

    # 1. Generate Hypotheses (now using informed sampling)
    hypotheses = generate_hypotheses(grid, num_hypotheses)

    # 2. Generate Map-Scans and Calculate CAER
    hypothesis_caers = []
    for h_x, h_y, h_theta in hypotheses:
        map_scan = simulate_lidar(h_x, h_y, h_theta, obstacles)  
        caer_value = caer(real_scan, map_scan)
        hypothesis_caers.append((caer_value, h_x, h_y, h_theta))

    # 3. Rank and Select Top-k Hypotheses
    hypothesis_caers.sort(key=lambda x: x[0])  
    top_k_hypotheses_caer = hypothesis_caers[:top_k]

    if screen is not None and robot_pos is not None and robot_theta is not None: 
        visualize_hypotheses(screen, top_k_hypotheses_caer, top_k, obstacles)
        pygame.display.flip()
        pygame.time.delay(2000)
        screen.fill((255, 255, 255)) 
        draw_obstacles(screen, obstacles)
        draw_goal(screen)
        draw_robot(screen, (robot_pos[0], robot_pos[1]), robot_theta) 
        pygame.display.flip()

    # 4. Scan-to-Map-Scan Matching (ICP)
    refined_poses = []
    for caer_val, x, y, theta in top_k_hypotheses_caer:
        map_scan = simulate_lidar(x, y, theta, obstacles)

        R, t, rmse = icp(np.array(real_scan), np.array(map_scan))

        refined_x = x + t[0] 
        refined_y = y + t[1]  
        refined_theta = theta + math.atan2(R[1, 0], R[0, 0])  

        refined_poses.append((rmse, refined_x, refined_y, refined_theta))

    refined_poses.sort(key=lambda x: x[0])  
    best_pose = refined_poses[0][1:]  
    return best_pose

# =============== MAIN SIMULATION ===============
def run_cbgl():
    pygame.init()
    screen = pygame.display.set_mode(MAP_SIZE)
    pygame.display.set_caption("CBGL Navigation with A*")
    clock = pygame.time.Clock()

    obstacles, grid = create_random_map()
    robot_pos = np.array([50.0, 50.0])  
    robot_theta = 0.0  
    ground_truth = np.array(robot_pos)
    map_scan = simulate_lidar(*ground_truth, robot_theta, obstacles)

    num_hypotheses = 1000  
    top_k = 10  
    localization_interval = 50  
    frame_count = 0

    start_cell = grid_from_pos(robot_pos)
    goal_cell = grid_from_pos(GOAL_POS)
    path_cells = a_star(grid, start_cell, goal_cell)
    path_points = deque([pos_from_grid(cell) for cell in path_cells])

    log = []
    trajectory_gt = []
    trajectory_est = []

    running = True
    while running:
        screen.fill((255, 255, 255))
        draw_obstacles(screen, obstacles)
        draw_goal(screen)
        draw_path(screen, path_cells)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        noisy_scan = simulate_lidar(robot_pos[0], robot_pos[1], robot_theta, obstacles)

        if frame_count % localization_interval == 0:
            estimated_x, estimated_y, estimated_theta = cbgl_localize(noisy_scan, obstacles, grid, num_hypotheses, top_k, screen, robot_pos, robot_theta)
            estimated_pose = np.array([estimated_x, estimated_y])
            pose_error = np.linalg.norm(robot_pos - estimated_pose) 
            theta_error = abs(robot_theta - estimated_theta)
            log.append((pose_error, theta_error))
            trajectory_est.append(estimated_pose.tolist())
        else:
            estimated_pose = robot_pos.copy()  
            trajectory_est.append(estimated_pose.tolist())

        trajectory_gt.append(robot_pos.copy().tolist())  
        if path_points:
            next_point = np.array(path_points[0])
            direction = next_point - estimated_pose  
            dist = np.linalg.norm(direction)
            if dist < 5:
                path_points.popleft()
            else:
                robot_theta = math.degrees(math.atan2(direction[1], direction[0]))
                step_x = robot_pos[0] + 2 * math.cos(math.radians(robot_theta))
                step_y = robot_pos[1] + 2 * math.sin(math.radians(robot_theta))
                if not check_collision(step_x, step_y, obstacles):
                    robot_pos[0] = step_x
                    robot_pos[1] = step_y
                else:
                    print("Collision! Replanning...")
                    start_cell = grid_from_pos(estimated_pose)  
                    path_cells = a_star(grid, start_cell, goal_cell)
                    path_points = deque([pos_from_grid(cell) for cell in path_cells])
        else:
            print("Goal Reached!")
            break

        draw_lidar(screen, robot_pos, noisy_scan)
        draw_robot(screen, robot_pos, robot_theta)
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1

    pygame.quit()

    log = np.array(log)
    if len(log) > 0:
        print("Average Pose Error:", np.mean(log[:, 0]))
        print("Average Theta Error:", np.mean(log[:, 1]))
        export_logs(log)

    trajectory_gt = np.array(trajectory_gt)
    trajectory_est = np.array(trajectory_est)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], label='Ground Truth', linewidth=2)
    plt.plot(trajectory_est[:, 0], trajectory_est[:, 1], label='Estimated', linestyle='--')
    plt.scatter(*GOAL_POS, color='orange', label='Goal')
    plt.title("Trajectory Comparison")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    steps = np.arange(len(log))
    if len(log) > 0:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(steps, log[:, 0], label='Pose Error', color='blue')
        plt.xlabel("Step")
        plt.ylabel("Pose Error")
        plt.title("Pose Error Over Time")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(steps, log[:, 1], label='Theta Error', color='red')
        plt.xlabel("Step")
        plt.ylabel("Theta Error")
        plt.title("Theta Error Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_cbgl()