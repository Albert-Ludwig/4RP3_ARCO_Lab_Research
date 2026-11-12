"""Inflated grid, shortest paths, and constant-speed path execution."""

import math
from functools import lru_cache

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from interfaces import NavigationResult, RobotState


class GridMap:
    def __init__(self, config):
        self.size = config["size_m"]
        self.resolution = config["resolution_m"]
        self.home = np.array(config["home_m"], float)
        self.radius = np.linalg.norm(config["robot_size_m"]) / 2 + config["margin_m"]
        self.obstacles = [Polygon(p) for p in config["obstacles"]]
        if len(self.obstacles) != 2 or any(not p.is_valid or p.area > 10 for p in self.obstacles):
            raise ValueError("Require two valid obstacles, each at most 10 m²")
        if self.obstacles[0].intersects(self.obstacles[1]):
            raise ValueError("Obstacles overlap")
        # Shapely buffers use inscribed arcs. Enlarge slightly to remain conservative.
        self.buffer_radius = self.radius / np.cos(np.pi / (4 * 16))
        self.inflated = unary_union([p.buffer(self.buffer_radius, quad_segs=16)
                                     for p in self.obstacles])
        self.boundary = box(self.radius, self.radius,
                            self.size - self.radius, self.size - self.radius)
        self.graph = nx.Graph()
        count = int(round(self.size / self.resolution)) + 1
        for i in range(count):
            for j in range(count):
                node = (i, j)
                if self.is_free(self.xy(node)):
                    self.graph.add_node(node)
        for i, j in list(self.graph):
            for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                other = (i + di, j + dj)
                if other not in self.graph:
                    continue
                if di and dj and ((i + di, j) not in self.graph or (i, j + dj) not in self.graph):
                    continue
                if self.segment_free(self.xy((i, j)), self.xy(other)):
                    self.graph.add_edge((i, j), other,
                                        weight=self.resolution * math.hypot(di, dj))
        home = self.node(self.home)
        if home not in self.graph:
            raise ValueError("Home is not navigable")
        self.graph = self.graph.subgraph(nx.node_connected_component(self.graph, home)).copy()
        self.free_nodes = sorted(self.graph.nodes)
        self.free_xy = np.array([self.xy(n) for n in self.free_nodes])

    def xy(self, node):
        return np.array(node, float) * self.resolution

    def node(self, xy):
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return None
        node = tuple(np.rint(xy / self.resolution).astype(int))
        return node if np.allclose(self.xy(node), xy, atol=1e-9, rtol=0) else None

    def is_free(self, xy):
        point = Point(xy)
        return self.boundary.covers(point) and not self.inflated.intersects(point)

    def segment_free(self, start, goal):
        if np.array_equal(start, goal):
            return self.is_free(start)
        line = LineString([start, goal])
        return self.boundary.covers(line) and not self.inflated.intersects(line)

    def is_reachable(self, xy):
        if not self.is_free(xy):
            return False
        lower = np.floor(np.asarray(xy) / self.resolution).astype(int)
        return any(tuple(lower + [i, j]) in self.graph
                   and self.segment_free(xy, self.xy(lower + [i, j]))
                   for i in (0, 1) for j in (0, 1))

    @lru_cache(maxsize=128)
    def paths_from(self, start):
        return nx.single_source_dijkstra(self.graph, start, weight="weight")

    def route(self, start, goal):
        a, b = self.node(start), self.node(goal)
        if a not in self.graph or b not in self.graph:
            return np.empty((0, 2)), np.inf
        distances, paths = self.paths_from(a)
        if b not in paths:
            return np.empty((0, 2)), np.inf
        return np.array([self.xy(n) for n in paths[b]]), distances[b]

    def candidates(self, count, seed):
        if not 2 <= count <= len(self.free_xy):
            raise ValueError("Candidate count exceeds reachable grid")
        rng = np.random.default_rng(seed)
        pool = self.free_xy[rng.permutation(len(self.free_xy))]
        chosen = [self.home.copy()]
        distance = np.full(len(pool), np.inf)
        for _ in range(count - 1):
            distance = np.minimum(distance, np.sum((pool - chosen[-1])**2, axis=1))
            chosen.append(pool[np.argmax(distance)].copy())
        return np.array(chosen)


class PythonNavigationSimulator:
    def __init__(self, grid, speed, step_s=1.0):
        if speed <= 0 or step_s <= 0:
            raise ValueError("Speed and timestep must be positive")
        self.grid, self.speed, self.step_s = grid, speed, step_s
        self.state = RobotState(*grid.home, 0.0, 0.0)
        self.trajectory = []
        self._record("start")

    def _record(self, status):
        self.trajectory.append({**vars(self.state), "status": status})

    def get_robot_state(self):
        return self.state

    def static_failure_reason(self, goal):
        """Return proof from the static map, or None when permanence is unknown.

        A future Nav2 adapter may provide the same hook. A local planner failure
        or timeout alone is not evidence that the static map has no route.
        """
        if not self.grid.is_free(goal):
            return "static_collision"
        # An off-grid request is unsupported by this backend, not proof that
        # the physical position has no path. Leave it temporary/unknown.
        if self.grid.node(goal) is None:
            return None
        path, distance = self.grid.route(self.grid.home, goal)
        if len(path) == 0 or not np.isfinite(distance):
            return "static_no_path"
        return None

    def estimate_travel_cost(self, start, goal):
        return self.grid.route(start, goal)[1] / self.speed

    def navigate_to(self, goal):
        path, distance = self.grid.route(self.state.xy, goal)
        if len(path) == 0 or not np.isfinite(distance):
            return NavigationResult("unreachable", self.state, 0.0, 0.0, path)
        before = self.state.elapsed_s
        for start, end in zip(path[:-1], path[1:]):
            if not self.grid.segment_free(start, end):
                return NavigationResult("blocked", self.state,
                                        (self.state.elapsed_s - before) * self.speed,
                                        self.state.elapsed_s - before, path)
            length = np.linalg.norm(end - start)
            duration = length / self.speed
            steps = max(1, math.ceil(duration / self.step_s))
            elapsed = self.state.elapsed_s
            yaw = float(np.arctan2(*(end - start)[::-1]))
            for step in range(1, steps + 1):
                xy = start + (end - start) * step / steps
                self.state = RobotState(*xy, yaw, elapsed + duration * step / steps)
                self._record("moving")
        return NavigationResult("arrived", self.state, float(distance),
                                self.state.elapsed_s - before, path)

    def wait(self, duration_s):
        if duration_s < 0:
            raise ValueError("Negative waiting time")
        s = self.state
        self.state = RobotState(s.x_m, s.y_m, s.yaw_rad, s.elapsed_s + duration_s)
        self._record("sampling")


def travel_matrix(nav, points):
    return np.array([[nav.estimate_travel_cost(a, b) for b in points] for a in points])
