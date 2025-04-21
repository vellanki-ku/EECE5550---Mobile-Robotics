import pygame
import numpy as np
import random
import heapq
from scipy.spatial import KDTree
from scipy.optimize import least_squares

MAP_SIZE = (100, 100)
CELL_SIZE = 5
NUM_OBSTACLES = 40
OBSTACLE_SIZE = 5
NUM_HYPOTHESES = 200
K_BEST_HYPOTHESES = 20
MAX_ITER_SCAN_MATCH = 10
LIDAR_RANGE = 10
LIDAR_ANGULAR_RES = np.pi / 180

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
CEMENT = (210, 210, 210)
OLIVE = (128, 128, 0)
BROWN = (139, 69, 19)

pygame.init()
SCREEN = pygame.display.set_mode((MAP_SIZE[0] * CELL_SIZE, MAP_SIZE[1] * CELL_SIZE))
pygame.display.set_caption("CBGL Navigation")
CLOCK = pygame.time.Clock()


class Map:
    def __init__(self, size, num_obstacles, obstacle_size):
        self.size = size
        self.grid = np.zeros(size, dtype=np.uint8)
        self.generate_obstacles(num_obstacles, obstacle_size)
        self.obstacle_positions = self.get_obstacle_positions()

    def generate_obstacles(self, num_obstacles, obstacle_size):
        for _ in range(num_obstacles):
            x = random.randint(0, self.size[0] - obstacle_size - 1)
            y = random.randint(0, self.size[1] - obstacle_size - 1)
            self.grid[x:x + obstacle_size, y:y + obstacle_size] = 1

    def is_free(self, x, y):
        return 0 <= x < self.size[0] and 0 <= y < self.size[1] and self.grid[x, y] == 0

    def draw(self, screen):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.grid[x, y] == 1:
                    pygame.draw.rect(screen, BLACK,
                                     (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def get_obstacle_positions(self):
        return [(x, y) for x in range(self.size[0]) for y in range(self.size[1]) if
                self.grid[x, y] == 1]

    def get_map_array(self):
        return self.grid.copy()


class Node:
    def __init__(self, x, y, cost, priority):
        self.x, self.y = x, y
        self.cost = cost
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


def astar(grid, start, goal):
    w, h = grid.shape
    open_set = []
    heapq.heappush(open_set, Node(start[0], start[1], 0, 0))
    came_from = {}
    cost_so_far = {start: 0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        current = heapq.heappop(open_set)
        if (current.x, current.y) == goal:
            path = []
            while (current.x, current.y) != start:
                path.append((current.x, current.y))
                current.x, current.y = came_from[(current.x, current.y)]
            path.reverse()
            return path

        for dx, dy in dirs:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[nx, ny] == 0:
                new_cost = cost_so_far[(current.x, current.y)] + 1.4 if dx * dy != 0 else \
                    cost_so_far[(current.x, current.y)] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(goal[0] - nx) + abs(goal[1] - ny)
                    heapq.heappush(open_set, Node(nx, ny, new_cost, priority))
                    came_from[(nx, ny)] = (current.x, current.y)
    return []


class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path = []
        self.theta = 0.0
        self.goal = None
        self.estimated_path = []  

    def update(self):
        if self.path:
            self.x, self.y = self.path.pop(0)
            if self.path:
                dx = self.path[0][0] - self.x
                dy = self.path[0][1] - self.y
                self.theta = np.arctan2(dy, dx)
            self.estimated_path.append((self.x, self.y))  

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE,
                         (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2),
                         CELL_SIZE // 2)
        pygame.draw.line(screen, BLUE,
                         (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2),
                         (self.x * CELL_SIZE + CELL_SIZE // 2 + 10 * np.cos(self.theta),
                          self.y * CELL_SIZE + CELL_SIZE // 2 + 10 * np.sin(self.theta)), 2)

    def get_pose(self):
        return (self.x, self.y, self.theta)

    def set_pose(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def draw_estimated_path(self, screen):
        if self.estimated_path:
            for p in self.estimated_path:
                pygame.draw.rect(screen, ORANGE, (p[0] * CELL_SIZE, p[1] * CELL_SIZE, 2, 2))  


def generate_hypotheses(grid, num_hypotheses):
    hypotheses = []
    free_cells = np.where(grid == 0)
    free_indices = list(zip(free_cells[0], free_cells[1]))

    if len(free_indices) < num_hypotheses:
        raise ValueError("Not enough free cells in the map to generate the required number of hypotheses.")

    random_indices = random.sample(free_indices, num_hypotheses)

    for x, y in random_indices:
        theta = random.uniform(-np.pi, np.pi)
        hypotheses.append((x, y, theta))
    return hypotheses


def generate_map_scan(grid, pose, lidar_range, lidar_angular_res):
    scan = []
    x, y, theta = pose
    for angle in np.arange(-np.pi, np.pi, lidar_angular_res):
        ray_x = x + 0.5
        ray_y = y + 0.5
        for _ in range(lidar_range * 2):
            ray_x += np.cos(theta + angle) * 0.5
            ray_y += np.sin(theta + angle) * 0.5
            ray_x_int = int(np.floor(ray_x))
            ray_y_int = int(np.floor(ray_y))

            if 0 <= ray_x_int < grid.shape[0] and 0 <= ray_y_int < grid.shape[1]:
                if grid[ray_x_int, ray_y_int] == 1:
                    scan.append((ray_x, ray_y))
                    break
            else:
                scan.append((ray_x, ray_y))
                break
        else:
            scan.append((ray_x, ray_y))
    return scan


def calculate_caer(real_scan, map_scan):
    if not real_scan or not map_scan:
        return float('inf')

    real_tree = KDTree(real_scan)
    distances = []
    for point in map_scan:
        dist, _ = real_tree.query(point, k=1)
        distances.append(dist)
    return np.mean(distances)


def scan_matching(real_scan, map_scan, initial_pose):
    def error_function(transform_params):
        dx, dy, dtheta = transform_params
        transformed_scan = [(x + dx, y + dy) for x, y in map_scan]
        transformed_scan = [(
            x * np.cos(dtheta) - y * np.sin(dtheta),
            x * np.sin(dtheta) + y * np.cos(dtheta)
        ) for x, y in transformed_scan]

        real_tree = KDTree(real_scan)
        distances = []
        for point in transformed_scan:
            dist, _ = real_tree.query(point)
            distances.append(dist)
        return np.array(distances)

    initial_params = [0.0, 0.0, 0.0]
    result = least_squares(error_function, initial_params, max_nfev=MAX_ITER_SCAN_MATCH)
    dx, dy, dtheta = result.x
    refined_x = initial_pose[0] + dx
    refined_y = initial_pose[1] + dy
    refined_theta = initial_pose[2] + dtheta
    return (refined_x, refined_y, refined_theta)


def refine_pose_estimate(grid, real_scan, hypotheses, k, lidar_range, lidar_angular_res):
    caer_values = []
    for hypothesis in hypotheses:
        map_scan = generate_map_scan(grid, hypothesis, lidar_range, lidar_angular_res)
        caer = calculate_caer(real_scan, map_scan)
        caer_values.append(caer)

    ranked_hypotheses = sorted(zip(hypotheses, caer_values), key=lambda x: x[1])
    best_hypotheses = [h for h, _ in ranked_hypotheses[:k]]

    refined_poses = []
    for pose in best_hypotheses:
        map_scan = generate_map_scan(grid, pose, lidar_range, lidar_angular_res)
        refined_pose = scan_matching(real_scan, map_scan, pose)
        refined_poses.append(refined_pose)

    return refined_poses


def get_real_scan(robot_pose, grid, lidar_range, lidar_angular_res):
    return generate_map_scan(grid, robot_pose, lidar_range, lidar_angular_res)


def draw_lidar_scan(screen, scan, pose):
    if scan:
        robot_x_px = pose[0] * CELL_SIZE + CELL_SIZE // 2
        robot_y_px = pose[1] * CELL_SIZE + CELL_SIZE // 2
        for x, y in scan:
            pygame.draw.line(screen, CEMENT, (robot_x_px, robot_y_px),
                             (x * CELL_SIZE, y * CELL_SIZE), 1)
            pygame.draw.circle(screen, CEMENT, (x * CELL_SIZE, y * CELL_SIZE), 1)  


def draw_pose(screen, pose, color):
    x_px = pose[0] * CELL_SIZE + CELL_SIZE // 2
    y_px = pose[1] * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, color, (x_px, y_px), 4)
    pygame.draw.line(screen, color, (x_px, y_px),
                     (x_px + 8 * np.cos(pose[2]), y_px + 8 * np.sin(pose[2])), 2)


def move_towards_goal(robot, grid, astar_path):
    if astar_path:
        robot.path = astar_path
        robot.update()
        if (int(robot.x), int(robot.y)) == (int(robot.goal[0]), int(robot.goal[1])):
            return []
        return astar_path
    else:
        return []


def random_free_point(grid):
    free_cells = np.where(grid == 0)
    free_indices = list(zip(free_cells[0], free_cells[1]))
    if free_indices:
        return random.choice(free_indices)
    return None  

def draw_pose(screen, pose, color=RED):
    x, y, theta = pose
    center = (int(x * CELL_SIZE + CELL_SIZE // 2), int(y * CELL_SIZE + CELL_SIZE // 2))
    end = (
        int(center[0] + 10 * np.cos(theta)),
        int(center[1] + 10 * np.sin(theta))
    )
    pygame.draw.circle(screen, color, center, CELL_SIZE // 2)
    pygame.draw.line(screen, color, center, end, 2)


if __name__ == '__main__':
    map_obj = Map(MAP_SIZE, NUM_OBSTACLES, OBSTACLE_SIZE)
    start = random_free_point(map_obj.grid)
    goal = random_free_point(map_obj.grid)

    if not start or not goal:
        raise ValueError("Could not find suitable start or goal positions.")

    robot = Robot(*start)
    robot.set_goal(goal)

    path = astar(map_obj.grid, start, goal)

    hypotheses = generate_hypotheses(map_obj.grid, NUM_HYPOTHESES)
    real_scan = get_real_scan(robot.get_pose(), map_obj.grid, LIDAR_RANGE,
                              LIDAR_ANGULAR_RES)
    best_hypotheses = refine_pose_estimate(map_obj.grid, real_scan, hypotheses,
                                            K_BEST_HYPOTHESES, LIDAR_RANGE,
                                            LIDAR_ANGULAR_RES)

    estimated_pose = best_hypotheses[0] if best_hypotheses else robot.get_pose()
    robot.set_pose(*estimated_pose)
    robot.estimated_path.append((robot.x, robot.y))  

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill(WHITE)
        map_obj.draw(SCREEN)

        pygame.draw.circle(SCREEN, GREEN,
                         (start[0] * CELL_SIZE + CELL_SIZE // 2,
                          start[1] * CELL_SIZE + CELL_SIZE // 2), 5)
        pygame.draw.circle(SCREEN, RED,
                         (goal[0] * CELL_SIZE + CELL_SIZE // 2,
                          goal[1] * CELL_SIZE + CELL_SIZE // 2), 5)

        for h in hypotheses:
            draw_pose(SCREEN, h, OLIVE)

        for h in best_hypotheses:
            draw_pose(SCREEN, h, BROWN)

        if path:
            for p in path:
                pygame.draw.rect(SCREEN, YELLOW,
                                 (p[0] * CELL_SIZE, p[1] * CELL_SIZE, CELL_SIZE,
                                  CELL_SIZE))

        if path:
            path = move_towards_goal(robot, map_obj.grid, path)  
        robot.draw(SCREEN)
        robot.draw_estimated_path(SCREEN) 

        real_scan = get_real_scan(robot.get_pose(), map_obj.grid, LIDAR_RANGE,
                                        LIDAR_ANGULAR_RES)
        draw_lidar_scan(SCREEN, real_scan, robot.get_pose())

        pygame.display.flip()
        CLOCK.tick(30)

    pygame.quit()