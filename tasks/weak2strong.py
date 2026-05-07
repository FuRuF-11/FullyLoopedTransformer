
import os
import random
import numpy as np
from mazelib import Maze
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
from sudoku import Sudoku

def Prims_maze(H: int = 10, W: int = 10):
    """
    Prim maze generator
    """
    if H % 2 == 0:
        H -= 1
    if W % 2 == 0:
        W -= 1
    grid = np.ones((H, W), dtype=np.int8)
    start_r = random.randrange(1, H, 2)
    start_c = random.randrange(1, W, 2)
    grid[start_r, start_c] = 0
    frontier = []
    in_frontier = set()
    def add_frontier(r, c):
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            wr, wc = r + dr // 2, c + dc // 2  
            if (1 <= nr < H - 1 and 1 <= nc < W - 1 and 
                grid[nr, nc] == 1 and (nr, nc) not in in_frontier):
                frontier.append((wr, wc, nr, nc))
                in_frontier.add((nr, nc))
    add_frontier(start_r, start_c)
    while frontier:
        idx = random.randrange(len(frontier))
        wr, wc, nr, nc = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if grid[nr, nc] == 1:
            grid[wr, wc] = 0     
            grid[nr, nc] = 0     
            add_frontier(nr, nc) 
    return grid

def maza_maker(
    h: int=10,
    w: int=10
    ):
    Maze.set_seed(123)
    m = Maze()
    # prims make sure the solution is only.
    m.grid = Prims_maze(h,w)
    m.generate_entrances()
    m.solver = BacktrackingSolver()
    m.solve()

    # puzzle solution
    return  m.tostring(True, False), m.tostring(True, True)

def sudoku_maker(
    n: int =10,
    level: str="hard"
):
    puzzle = Sudoku(2,4).difficulty(0.3)
    if puzzle.has_multiple_solutions():
        problem=puzzle.board
        # puzzle solution
        solution=puzzle.solve()
        return problem, solution

def main():
    # maza_maker(15,27)
    sudoku_maker(6)


if __name__=="__main__":
    main()
