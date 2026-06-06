from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import sys


# -----------------------------
# Data model
# -----------------------------
@dataclass(order=True)
class State:
    sort_index: Tuple[int, int, int] = field(init=False, repr=False)
    row: int
    col: int
    g_cost: int
    h_cost: int
    path: List[Tuple[int, int]]

    def __post_init__(self):
        # Sorting priority:
        # 1) Lower heuristic
        # 2) Lower traversal cost
        # 3) Smaller path length
        self.sort_index = (self.h_cost, self.g_cost, len(self.path))

    @property
    def position(self) -> Tuple[int, int]:
        return (self.row, self.col)

    @property
    def f_cost(self) -> int:
        return self.g_cost + self.h_cost


# -----------------------------
# Local Beam Search Optimizer
# -----------------------------
class RouteOptimizer:
    def __init__(self, grid: List[List[str]], k: int = 2):
        if not grid or not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("Grid must be non-empty and rectangular.")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.k = k

        self.start = None
        self.goal = None
        self.high_cost_cells = set()

        self._locate_special_cells()

    def _locate_special_cells(self):
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell == 'S':
                    self.start = (r, c)
                elif cell == 'G':
                    self.goal = (r, c)
                elif cell == 'C':
                    self.high_cost_cells.add((r, c))

        if self.start is None:
            raise ValueError("Start state 'S' not found in grid.")
        if self.goal is None:
            raise ValueError("Goal state 'G' not found in grid.")

    def heuristic(self, row: int, col: int) -> int:
        gr, gc = self.goal
        return abs(gr - row) + abs(gc - col)

    def is_valid(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] != 'X'

    def move_cost(self, row: int, col: int) -> int:
        # Normal move cost = 1
        # Additional penalty for C cell = 2
        base = 1
        penalty = 2 if (row, col) in self.high_cost_cells else 0
        return base + penalty

    def generate_successors(self, state: State) -> List[State]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        successors = []

        for dr, dc in directions:
            nr, nc = state.row + dr, state.col + dc

            if not self.is_valid(nr, nc):
                continue

            # Avoid immediate cycles in the current path
            if (nr, nc) in state.path:
                continue

            step_cost = self.move_cost(nr, nc)
            new_g = state.g_cost + step_cost
            new_h = self.heuristic(nr, nc)
            new_path = state.path + [(nr, nc)]

            successors.append(State(nr, nc, new_g, new_h, new_path))

        return successors

    def remove_duplicates_keep_best(self, states: List[State]) -> List[State]:
        best = {}
        for st in states:
            pos = st.position
            if pos not in best:
                best[pos] = st
            else:
                # Keep better state based on h, then g, then shorter path
                if st.sort_index < best[pos].sort_index:
                    best[pos] = st
        return list(best.values())

    def local_beam_search(self) -> Tuple[Optional[State], List[List[State]]]:
        sr, sc = self.start
        start_state = State(
            row=sr,
            col=sc,
            g_cost=0,
            h_cost=self.heuristic(sr, sc),
            path=[(sr, sc)]
        )

        beam = [start_state]
        visited_beams = [beam[:]]

        iteration = 0
        max_iterations = self.rows * self.cols * 10

        while iteration < max_iterations:
            iteration += 1

            # Goal test in current beam
            for st in beam:
                if st.position == self.goal:
                    return st, visited_beams

            all_successors = []
            for st in beam:
                all_successors.extend(self.generate_successors(st))

            if not all_successors:
                return None, visited_beams

            all_successors = self.remove_duplicates_keep_best(all_successors)
            all_successors.sort()

            beam = all_successors[:self.k]
            visited_beams.append(beam[:])

        return None, visited_beams

    def print_beam_history(self, history: List[List[State]]):
        print("\n=== Local Beam Search Execution Flow ===")
        for i, beam in enumerate(history):
            print(f"\nIteration {i}:")
            for j, st in enumerate(beam, start=1):
                print(
                    f"  Beam {j}: Position={st.position}, "
                    f"Heuristic={st.h_cost}, Traversal Cost={st.g_cost}, "
                    f"f(n)={st.f_cost}, Path={st.path}"
                )

    def overlay_path_on_grid(self, path: List[Tuple[int, int]]) -> List[List[str]]:
        display = [row[:] for row in self.grid]
        for r, c in path:
            if display[r][c] not in ('S', 'G'):
                display[r][c] = '*'
        return display

    def print_grid(self, grid_to_print: List[List[str]]):
        print("\nGrid:")
        for row in grid_to_print:
            print(" ".join(row))


# -----------------------------
# Input utilities
# -----------------------------
def read_grid_from_file(filename: str) -> List[List[str]]:
    """
    Expected simple input format:
    Each line represents one row.
    Allowed symbols: S, G, X, C, .
    Example:
    S C . X .
    . X . C .
    . X . . .
    C . X X .
    . . C G .

    You may also use spaces or commas; both are supported.
    """
    grid = []

    try:
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{filename}' not found.")

    for line in lines:
        line = line.replace(",", " ")
        tokens = line.split()

        # Support compact format like SC.X.
        if len(tokens) == 1 and len(tokens[0]) > 1:
            row = list(tokens[0])
        else:
            row = tokens

        grid.append(row)

    return grid


def write_output_to_file(filename: str, result_text: str):
    try:
        with open(filename, "w") as f:
            f.write(result_text)
    except Exception as e:
        print(f"Error writing output file: {e}")


# -----------------------------
# Default sample grid from assignment
# -----------------------------
def assignment_sample_grid() -> List[List[str]]:
    # Based on the assignment figure, represented as a 5x5 grid.
    return [
        ['S', '.', 'C', 'X', '.'],
        ['.', 'X', '.', 'C', '.'],
        ['.', 'X', '.', '.', '.'],
        ['C', '.', 'X', 'X', '.'],
        ['.', '.', 'C', 'G', '.']
    ]


# -----------------------------
# Main runner
# -----------------------------
def solve_route_optimizer(grid: List[List[str]], k: int = 2) -> str:
    optimizer = RouteOptimizer(grid, k=k)
    goal_state, history = optimizer.local_beam_search()

    output_lines = []
    output_lines.append("=== Route Optimizer using Local Beam Search ===")
    output_lines.append(f"Beam width (k): {k}")
    output_lines.append(f"Start state: {optimizer.start}")
    output_lines.append(f"Goal state: {optimizer.goal}")
    output_lines.append("")

    output_lines.append("Execution Flow:")
    for i, beam in enumerate(history):
        output_lines.append(f"Iteration {i}:")
        for j, st in enumerate(beam, start=1):
            output_lines.append(
                f"  Beam {j}: Position={st.position}, "
                f"Heuristic={st.h_cost}, Traversal Cost={st.g_cost}, "
                f"f(n)={st.f_cost}, Path={st.path}"
            )
        output_lines.append("")

    if goal_state is None:
        output_lines.append("Goal not reachable using Local Beam Search.")
        return "\n".join(output_lines)

    output_lines.append("Final Result:")
    output_lines.append(f"Goal reached at: {goal_state.position}")
    output_lines.append(f"Final path: {goal_state.path}")
    output_lines.append(f"Total path cost: {len(goal_state.path) - 1}")
    output_lines.append(f"Total optimized traversal cost: {goal_state.g_cost}")

    path_grid = optimizer.overlay_path_on_grid(goal_state.path)
    output_lines.append("\nPath on Grid:")
    for row in path_grid:
        output_lines.append(" ".join(row))

    return "\n".join(output_lines)


def main():
    input_file = "inputPS11.txt"
    output_file = "outputPS11.txt"

    try:
        # If file exists and works, use it; otherwise use sample grid.
        try:
            grid = read_grid_from_file(input_file)
        except Exception:
            print(f"Could not read '{input_file}'. Using assignment sample grid instead.")
            grid = assignment_sample_grid()

        result = solve_route_optimizer(grid, k=2)
        print(result)
        write_output_to_file(output_file, result)

    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        write_output_to_file(output_file, error_message)


if __name__ == "__main__":
    main()