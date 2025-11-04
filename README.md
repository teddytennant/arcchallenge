# ARC Challenge Solver

A Python-based solver for the Abstraction and Reasoning Corpus (ARC) challenge.

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Structure

- `arc_core/`: Core grid and object operations
- `dsl/`: Domain-specific language for ARC programs
- `synth/`: Program synthesis engine
- `solvers/`: Specialized solvers for common motifs
- `scripts/`: Evaluation and utility scripts

## Usage

Run evaluation on dev tasks:
```bash
python scripts/eval.py
```