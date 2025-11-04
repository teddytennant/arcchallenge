#!/usr/bin/env python3
"""Simple evaluation script."""

import json
from pathlib import Path

def load_arc_data(path: str):
    """Load ARC tasks."""
    with open(path) as f:
        return json.load(f)

def evaluate():
    # Placeholder: assume data in data/arc-agi_evaluation_challenges.json
    data_path = Path(__file__).parent.parent / "data" / "arc-agi_evaluation_challenges.json"
    if not data_path.exists():
        print("ARC data not found. Download from https://github.com/fchollet/ARC")
        return
    
    tasks = load_arc_data(data_path)
    print(f"Loaded {len(tasks)} tasks")
    
    # Dummy evaluation
    solved = 0
    for task_id, task in tasks.items():
        # Try solvers
        from ..solvers.symmetry import solve_symmetry
        input_grids = [task['train'][0]['input']]  # simplify
        output_grids = [task['train'][0]['output']]
        result = solve_symmetry(input_grids, output_grids)
        if result is not None:
            solved += 1
    
    print(f"Solved {solved}/{len(tasks)} tasks")

if __name__ == "__main__":
    evaluate()