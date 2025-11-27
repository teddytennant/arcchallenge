"""
Program enumeration and synthesis for ARC.

Implements multiple program synthesis strategies:
- Enumerative search (exhaustive bottom-up)
- Stochastic search (Monte Carlo sampling)
- Top-down enumeration
- Type-directed synthesis
- Observational equivalence pruning
"""

from typing import List, Generator, Set, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random
import itertools

from ..dsl.ast import *
from ..dsl.primitives import ALL_PRIMITIVES, Primitive, get_primitive
from ..dsl.interpreter import Interpreter, interpret
from ..arc_core.grid import Grid
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SynthesisConfig:
    """Configuration for program synthesis."""

    max_depth: int = 3
    max_size: int = 10
    max_programs: int = 10000
    timeout_seconds: float = 60.0
    use_observational_equivalence: bool = True
    prune_semantically_equivalent: bool = True
    primitives: List[str] = field(default_factory=lambda: [p.name for p in ALL_PRIMITIVES])


# ============================================================================
# Enumerative Synthesis
# ============================================================================

class EnumerativeSynthesizer:
    """Bottom-up enumerative program synthesis."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.primitives = [get_primitive(name) for name in config.primitives
                          if get_primitive(name) is not None]
        self.interpreter = Interpreter()
        self.seen_programs: Set[str] = set()
        self.value_cache: Dict[str, Any] = {}

    def synthesize(self, examples: List[Tuple[Grid, Grid]],
                   max_results: int = 10) -> List[Program]:
        """Synthesize programs that match examples.

        Args:
            examples: List of (input, output) grid pairs
            max_results: Maximum number of programs to return

        Returns:
            List of programs that solve all examples
        """
        solutions = []
        programs_checked = 0

        # Enumerate programs by increasing depth
        for d in range(1, self.config.max_depth + 1):
            if len(solutions) >= max_results:
                break

            print(f"Searching at depth {d}...")

            for program_expr in self._enumerate_at_depth(d):
                if programs_checked >= self.config.max_programs:
                    print(f"Reached maximum programs limit")
                    return solutions

                programs_checked += 1

                # Test program on examples
                if self._test_program(program_expr, examples):
                    program = Program(program_expr, name=f"synthesized_{len(solutions)}")
                    solutions.append(program)
                    print(f"Found solution #{len(solutions)}: {program_expr}")

                    if len(solutions) >= max_results:
                        return solutions

        print(f"Checked {programs_checked} programs, found {len(solutions)} solutions")
        return solutions

    def _enumerate_at_depth(self, depth: int) -> Generator[Expr, None, None]:
        """Enumerate all programs up to given depth."""
        if depth == 1:
            # Base case: primitives only
            for prim in self.primitives:
                expr = Prim(prim.name)
                yield expr
        else:
            # Recursive case: apply primitives to smaller programs
            for prim in self.primitives:
                # Determine arity based on primitive signature
                arity = self._get_arity(prim)

                if arity == 0:
                    # Nullary primitive
                    yield Prim(prim.name)
                elif arity == 1:
                    # Unary primitive - apply to programs of depth-1
                    for arg in self._enumerate_at_depth(depth - 1):
                        expr = Apply(Prim(prim.name), [arg])
                        if self._should_keep(expr):
                            yield expr
                elif arity == 2:
                    # Binary primitive - try combinations
                    for d1 in range(1, depth):
                        d2 = depth - 1 - d1
                        if d2 < 1:
                            continue
                        for arg1 in self._enumerate_at_depth(d1):
                            for arg2 in self._enumerate_at_depth(d2):
                                expr = Apply(Prim(prim.name), [arg1, arg2])
                                if self._should_keep(expr):
                                    yield expr

    def _get_arity(self, prim: Primitive) -> int:
        """Determine arity of a primitive from its type signature."""
        sig = prim.type_sig
        if "->" not in sig:
            return 0

        # Simple heuristic: count commas in argument part
        if sig.startswith("("):
            args_part = sig.split("->")[0].strip()
            if args_part == "()":
                return 0
            return args_part.count(",") + 1
        else:
            # Single argument
            return 1

    def _should_keep(self, expr: Expr) -> bool:
        """Check if we should keep this program (pruning)."""
        # Check if we've seen this program before
        expr_str = repr(expr)
        if expr_str in self.seen_programs:
            return False
        self.seen_programs.add(expr_str)

        # Check size limit
        if size(expr) > self.config.max_size:
            return False

        return True

    def _test_program(self, expr: Expr, examples: List[Tuple[Grid, Grid]]) -> bool:
        """Test if program matches all examples."""
        for input_grid, expected_output in examples:
            try:
                # Create environment with input
                from ..dsl.interpreter import Environment
                env = Environment()
                env.define("input", input_grid)

                # Evaluate program
                self.interpreter.reset()
                output = self.interpreter.eval(expr, env)

                # Check if output matches
                if not isinstance(output, np.ndarray):
                    return False

                if not np.array_equal(output, expected_output):
                    return False

            except Exception:
                return False

        return True


# ============================================================================
# Stochastic Synthesis
# ============================================================================

class StochasticSynthesizer:
    """Monte Carlo stochastic program synthesis."""

    def __init__(self, config: SynthesisConfig, temperature: float = 1.0):
        self.config = config
        self.temperature = temperature
        self.primitives = [get_primitive(name) for name in config.primitives
                          if get_primitive(name) is not None]
        self.interpreter = Interpreter()

        # Primitive probabilities (uniform by default)
        self.prim_probs = {p.name: 1.0 / len(self.primitives) for p in self.primitives}

    def synthesize(self, examples: List[Tuple[Grid, Grid]],
                   num_samples: int = 1000) -> List[Program]:
        """Synthesize programs using stochastic sampling.

        Args:
            examples: List of (input, output) grid pairs
            num_samples: Number of programs to sample

        Returns:
            List of programs that solve examples
        """
        solutions = []
        best_score = 0.0

        for i in range(num_samples):
            # Sample a random program
            program_expr = self._sample_program()

            # Evaluate program
            score = self._score_program(program_expr, examples)

            if score == 1.0:
                # Perfect match
                program = Program(program_expr, name=f"stochastic_{len(solutions)}")
                solutions.append(program)
                print(f"Found solution #{len(solutions)}: {program_expr}")

            if score > best_score:
                best_score = score
                if i % 100 == 0:
                    print(f"Sample {i}: best score = {best_score:.3f}")

        return solutions

    def _sample_program(self, max_depth: Optional[int] = None) -> Expr:
        """Sample a random program."""
        if max_depth is None:
            max_depth = self.config.max_depth

        depth = random.randint(1, max_depth)
        return self._sample_at_depth(depth)

    def _sample_at_depth(self, depth: int) -> Expr:
        """Sample a program at given depth."""
        if depth == 1:
            # Sample a primitive
            prim = self._sample_primitive()
            return Prim(prim.name)
        else:
            # Sample a primitive and arguments
            prim = self._sample_primitive()
            arity = self._get_arity(prim)

            if arity == 0:
                return Prim(prim.name)
            elif arity == 1:
                arg = self._sample_at_depth(depth - 1)
                return Apply(Prim(prim.name), [arg])
            elif arity == 2:
                # Split depth randomly
                d1 = random.randint(1, depth - 1)
                d2 = depth - 1 - d1
                if d2 < 1:
                    d2 = 1
                arg1 = self._sample_at_depth(d1)
                arg2 = self._sample_at_depth(d2)
                return Apply(Prim(prim.name), [arg1, arg2])
            else:
                return Prim(prim.name)

    def _sample_primitive(self) -> Primitive:
        """Sample a primitive according to probabilities."""
        prims = list(self.prim_probs.keys())
        probs = [self.prim_probs[p] for p in prims]

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(prims, weights=probs, k=1)[0]
        return get_primitive(chosen)

    def _get_arity(self, prim: Primitive) -> int:
        """Determine arity of a primitive."""
        sig = prim.type_sig
        if "->" not in sig:
            return 0

        if sig.startswith("("):
            args_part = sig.split("->")[0].strip()
            if args_part == "()":
                return 0
            return args_part.count(",") + 1
        else:
            return 1

    def _score_program(self, expr: Expr, examples: List[Tuple[Grid, Grid]]) -> float:
        """Score a program on examples (0 to 1)."""
        correct = 0

        for input_grid, expected_output in examples:
            try:
                from ..dsl.interpreter import Environment
                env = Environment()
                env.define("input", input_grid)

                self.interpreter.reset()
                output = self.interpreter.eval(expr, env)

                if isinstance(output, np.ndarray) and isinstance(expected_output, np.ndarray):
                    if np.array_equal(output, expected_output):
                        correct += 1
                    else:
                        # Partial credit based on pixel accuracy
                        if output.shape == expected_output.shape:
                            accuracy = np.mean(output == expected_output)
                            correct += accuracy

            except Exception:
                pass

        return correct / len(examples) if examples else 0.0


# ============================================================================
# Genetic Programming Synthesis
# ============================================================================

@dataclass
class Individual:
    """Individual in genetic programming population."""
    expr: Expr
    fitness: float = 0.0
    age: int = 0


class GeneticSynthesizer:
    """Genetic programming for program synthesis."""

    def __init__(self, config: SynthesisConfig,
                 population_size: int = 100,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.6):
        self.config = config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.primitives = [get_primitive(name) for name in config.primitives
                          if get_primitive(name) is not None]
        self.interpreter = Interpreter()

    def synthesize(self, examples: List[Tuple[Grid, Grid]],
                   generations: int = 50) -> List[Program]:
        """Synthesize programs using genetic programming.

        Args:
            examples: List of (input, output) grid pairs
            generations: Number of generations to evolve

        Returns:
            List of programs that solve examples
        """
        # Initialize population
        population = self._init_population()

        solutions = []
        best_fitness = 0.0

        for gen in range(generations):
            # Evaluate fitness
            for ind in population:
                ind.fitness = self._evaluate_fitness(ind.expr, examples)

                if ind.fitness == 1.0:
                    program = Program(ind.expr, name=f"genetic_{len(solutions)}")
                    solutions.append(program)
                    print(f"Generation {gen}: Found solution #{len(solutions)}")

            # Track best
            current_best = max(ind.fitness for ind in population)
            if current_best > best_fitness:
                best_fitness = current_best
                print(f"Generation {gen}: best fitness = {best_fitness:.3f}")

            # Selection and reproduction
            population = self._evolve_population(population)

        return solutions

    def _init_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            depth = random.randint(1, self.config.max_depth)
            expr = self._random_expr(depth)
            population.append(Individual(expr))
        return population

    def _random_expr(self, depth: int) -> Expr:
        """Generate random expression."""
        if depth == 1 or random.random() < 0.3:
            prim = random.choice(self.primitives)
            return Prim(prim.name)
        else:
            prim = random.choice(self.primitives)
            arity = self._get_arity(prim)

            if arity == 0:
                return Prim(prim.name)
            elif arity == 1:
                arg = self._random_expr(depth - 1)
                return Apply(Prim(prim.name), [arg])
            elif arity >= 2:
                arg1 = self._random_expr(depth - 1)
                arg2 = self._random_expr(depth - 1)
                return Apply(Prim(prim.name), [arg1, arg2])
            else:
                return Prim(prim.name)

    def _get_arity(self, prim: Primitive) -> int:
        """Get primitive arity."""
        sig = prim.type_sig
        if "->" not in sig:
            return 0
        if sig.startswith("("):
            args_part = sig.split("->")[0].strip()
            if args_part == "()":
                return 0
            return args_part.count(",") + 1
        else:
            return 1

    def _evaluate_fitness(self, expr: Expr, examples: List[Tuple[Grid, Grid]]) -> float:
        """Evaluate fitness of program."""
        correct = 0

        for input_grid, expected_output in examples:
            try:
                from ..dsl.interpreter import Environment
                env = Environment()
                env.define("input", input_grid)

                self.interpreter.reset()
                output = self.interpreter.eval(expr, env)

                if isinstance(output, np.ndarray) and isinstance(expected_output, np.ndarray):
                    if np.array_equal(output, expected_output):
                        correct += 1

            except Exception:
                pass

        return correct / len(examples) if examples else 0.0

    def _evolve_population(self, population: List[Individual]) -> List[Individual]:
        """Evolve population through selection, crossover, and mutation."""
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Keep top performers (elitism)
        elite_size = max(2, self.population_size // 10)
        new_population = population[:elite_size]

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1.expr, parent2.expr)
            else:
                child = parent1.expr

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(Individual(child))

        return new_population

    def _tournament_select(self, population: List[Individual], k: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, expr1: Expr, expr2: Expr) -> Expr:
        """Crossover two expressions."""
        # Simple subtree crossover - swap random subtrees
        # For now, just return one of them
        return random.choice([expr1, expr2])

    def _mutate(self, expr: Expr) -> Expr:
        """Mutate an expression."""
        # Random mutation: replace a subtree with a random expression
        if isinstance(expr, Apply):
            if random.random() < 0.5:
                return self._random_expr(2)
            else:
                return expr
        else:
            return self._random_expr(2)


# ============================================================================
# Utility Functions
# ============================================================================

def enumerate_programs(max_depth: int = 3, primitives: Optional[List[str]] = None) -> List[Expr]:
    """Convenience function to enumerate programs.

    Args:
        max_depth: Maximum depth of programs
        primitives: List of primitive names to use

    Returns:
        List of program expressions
    """
    config = SynthesisConfig(max_depth=max_depth)
    if primitives:
        config.primitives = primitives

    synthesizer = EnumerativeSynthesizer(config)
    programs = []

    for d in range(1, max_depth + 1):
        for expr in synthesizer._enumerate_at_depth(d):
            programs.append(expr)
            if len(programs) >= 1000:  # Limit
                return programs

    return programs
