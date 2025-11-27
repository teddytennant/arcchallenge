#!/usr/bin/env python3
"""
Comprehensive evaluation framework for ARC solver.

Usage:
    python scripts/eval.py --dataset data/training.json
    python scripts/eval.py --dataset data/evaluation.json --timeout 120
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_core.grid import load_grid, save_grid
from synth.enumerator import (
    EnumerativeSynthesizer,
    StochasticSynthesizer,
    GeneticSynthesizer,
    SynthesisConfig
)
from solvers import symmetry, tiling, object_tracking


def load_arc_dataset(path: str) -> Dict:
    """Load ARC dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_task(task_id: str, task_data: Dict, config: Dict) -> Dict:
    """
    Evaluate a single task.

    Args:
        task_id: Task identifier
        task_data: Task data with train and test examples
        config: Evaluation configuration

    Returns:
        Evaluation results
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating task: {task_id}")
    print(f"{'=' * 60}")

    train_examples = task_data['train']
    test_examples = task_data['test']

    # Convert to numpy arrays
    train_pairs = [
        (load_grid(ex['input']), load_grid(ex['output']))
        for ex in train_examples
    ]

    results = {
        'task_id': task_id,
        'solved': False,
        'method': None,
        'time': 0.0,
        'accuracy': 0.0,
        'program': None
    }

    start_time = time.time()

    # Try specialized solvers first (fast)
    print("Trying specialized solvers...")

    solvers_to_try = [
        ('symmetry', symmetry.solve_symmetry),
        ('tiling', tiling.solve_by_tiling),
        ('object_tracking', object_tracking.solve_by_object_tracking),
    ]

    for solver_name, solver_fn in solvers_to_try:
        try:
            transform_fn = solver_fn(train_pairs)
            if transform_fn is not None:
                print(f"  {solver_name} solver found a pattern!")

                # Test on test examples
                correct = 0
                for test_ex in test_examples:
                    test_input = load_grid(test_ex['input'])
                    expected_output = load_grid(test_ex['output'])

                    try:
                        predicted_output = transform_fn(test_input)
                        import numpy as np
                        if np.array_equal(predicted_output, expected_output):
                            correct += 1
                    except Exception as e:
                        print(f"  Error applying transformation: {e}")

                accuracy = correct / len(test_examples) if test_examples else 0

                if accuracy == 1.0:
                    elapsed = time.time() - start_time
                    results.update({
                        'solved': True,
                        'method': solver_name,
                        'time': elapsed,
                        'accuracy': accuracy
                    })
                    print(f"  ✓ SOLVED by {solver_name} in {elapsed:.2f}s")
                    return results
        except Exception as e:
            print(f"  {solver_name} solver failed: {e}")

    # Try program synthesis
    if config.get('use_synthesis', True):
        print("Trying program synthesis...")

        synth_config = SynthesisConfig(
            max_depth=config.get('max_depth', 3),
            max_programs=config.get('max_programs', 5000),
            timeout_seconds=config.get('timeout', 60)
        )

        synthesizer_type = config.get('synthesizer', 'enumerative')

        try:
            if synthesizer_type == 'enumerative':
                synthesizer = EnumerativeSynthesizer(synth_config)
            elif synthesizer_type == 'stochastic':
                synthesizer = StochasticSynthesizer(synth_config)
            elif synthesizer_type == 'genetic':
                synthesizer = GeneticSynthesizer(synth_config)
            else:
                print(f"Unknown synthesizer: {synthesizer_type}")
                return results

            # Run synthesis
            solutions = synthesizer.synthesize(train_pairs, max_results=1)

            if solutions:
                program = solutions[0]
                print(f"  Found program: {program.expr}")

                # Test on test examples
                from dsl.interpreter import Interpreter, Environment
                interpreter = Interpreter()

                correct = 0
                for test_ex in test_examples:
                    test_input = load_grid(test_ex['input'])
                    expected_output = load_grid(test_ex['output'])

                    try:
                        env = Environment()
                        env.define("input", test_input)
                        predicted_output = interpreter.eval(program.expr, env)

                        import numpy as np
                        if np.array_equal(predicted_output, expected_output):
                            correct += 1
                    except Exception as e:
                        print(f"  Error evaluating program: {e}")

                accuracy = correct / len(test_examples) if test_examples else 0

                elapsed = time.time() - start_time
                results.update({
                    'solved': accuracy == 1.0,
                    'method': f'synthesis_{synthesizer_type}',
                    'time': elapsed,
                    'accuracy': accuracy,
                    'program': str(program.expr)
                })

                if accuracy == 1.0:
                    print(f"  ✓ SOLVED by synthesis in {elapsed:.2f}s")
                else:
                    print(f"  Partial solution ({accuracy:.1%} accuracy)")

                return results

        except Exception as e:
            print(f"  Synthesis failed: {e}")

    elapsed = time.time() - start_time
    results['time'] = elapsed
    print(f"  ✗ UNSOLVED after {elapsed:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ARC solver')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to ARC dataset JSON')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Timeout per task in seconds')
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Maximum program depth for synthesis')
    parser.add_argument('--synthesizer', type=str, default='enumerative',
                       choices=['enumerative', 'stochastic', 'genetic'],
                       help='Synthesis strategy to use')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='Maximum number of tasks to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results JSON')
    parser.add_argument('--no-synthesis', action='store_true',
                       help='Disable program synthesis (only use pattern solvers)')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_arc_dataset(args.dataset)
    print(f"Loaded {len(dataset)} tasks")

    # Configuration
    config = {
        'timeout': args.timeout,
        'max_depth': args.max_depth,
        'synthesizer': args.synthesizer,
        'use_synthesis': not args.no_synthesis,
        'max_programs': 10000
    }

    # Evaluate tasks
    results = []
    task_ids = list(dataset.keys())[:args.max_tasks] if args.max_tasks else list(dataset.keys())

    for i, task_id in enumerate(task_ids):
        print(f"\n[{i+1}/{len(task_ids)}]")
        task_data = dataset[task_id]
        result = evaluate_task(task_id, task_data, config)
        results.append(result)

    # Compute statistics
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
    solved = sum(1 for r in results if r['solved'])
    avg_time = sum(r['time'] for r in results) / total if total > 0 else 0
    avg_accuracy = sum(r['accuracy'] for r in results) / total if total > 0 else 0

    by_method = defaultdict(int)
    for r in results:
        if r['solved']:
            by_method[r['method']] += 1

    print(f"Total tasks:     {total}")
    print(f"Solved:          {solved} ({solved/total*100:.1f}%)")
    print(f"Average time:    {avg_time:.2f}s")
    print(f"Average accuracy: {avg_accuracy:.1%}")
    print()
    print("Solutions by method:")
    for method, count in sorted(by_method.items()):
        print(f"  {method:20s}: {count}")

    # Save results
    if args.output:
        output_data = {
            'config': config,
            'summary': {
                'total': total,
                'solved': solved,
                'accuracy': solved / total if total > 0 else 0,
                'avg_time': avg_time,
                'by_method': dict(by_method)
            },
            'results': results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
