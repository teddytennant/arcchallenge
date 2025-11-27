# Architecture Documentation

This document provides an in-depth look at the architecture of the ARC Challenge Solver.

## System Overview

The solver is organized into several key modules, each with specific responsibilities:

```
┌──────────────────────────────────────────────────────────────┐
│                        ARC Solver                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  arc_core   │───▶│     dsl     │───▶│    synth    │      │
│  │             │    │             │    │             │      │
│  │ Grid & Obj  │    │ Primitives  │    │  Synthesis  │      │
│  │ Operations  │    │ Interpreter │    │  Algorithms │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                   │                   │            │
│         │                   │                   │            │
│         └───────────────────┴───────────────────┘            │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────┐                           │
│                    │   solvers   │                           │
│                    │             │                           │
│                    │ Specialized │                           │
│                    │   Patterns  │                           │
│                    └─────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. `arc_core` - Core Operations

The foundation layer providing grid and object primitives.

#### `arc_core/grid.py`

Implements 40+ grid operations:

**Categories:**
- **I/O**: `load_grid`, `save_grid`, `create_grid`
- **Spatial**: `rotate_90/180/270`, `reflect_*`, `scale_grid`, `tile_grid`
- **Color**: `remap_colors`, `swap_colors`, `canonicalize_colors`
- **Analysis**: `detect_symmetry`, `detect_period`, `is_tiled`
- **Components**: `connected_components`
- **Comparison**: `grids_equal`, `hamming_distance`

**Key Design Decisions:**
- All grids are numpy arrays with dtype `int8` (values 0-9)
- Immutable operations (return new grids, don't modify in place)
- Optional parameters with sensible defaults
- Comprehensive type hints

#### `arc_core/objects.py`

Object-oriented representation of grid regions:

**Object Class:**
```python
@dataclass
class Object:
    positions: Set[Tuple[int, int]]  # Cell positions
    color: int                        # Object color
    bbox: Tuple[int, int, int, int]   # Bounding box
    area: int                         # Number of cells
    centroid: Tuple[float, float]     # Center of mass
    shape_type: ShapeType             # Classified shape
    # ... more properties
```

**Features:**
- Automatic property computation via `__post_init__`
- Shape classification (rectangle, square, line, cross, etc.)
- Spatial relationship detection
- Object transformations (translate, rotate, scale)
- Hole counting (topological feature)

### 2. `dsl` - Domain-Specific Language

Type-safe DSL for expressing ARC transformations.

#### `dsl/primitives.py`

**Primitive Class:**
```python
class Primitive:
    name: str                # e.g., "rotate_90"
    impl: Callable           # Implementation function
    type_sig: str            # e.g., "Grid -> Grid"
    description: str         # Human-readable description
```

**Categories (60+ primitives):**
- Grid transformations (rotate, reflect, scale, tile)
- Object operations (extract, filter, transform)
- Aggregations (argmax, argmin, count, sort)
- Logic (and, or, not, if-then-else)
- Arithmetic (add, sub, mul, div, mod)

**Design:**
- First-class functions
- Partial application support
- Type signatures for synthesis guidance
- Registry pattern for extensibility

#### `dsl/ast.py`

**AST Node Types:**
```python
Var        # Variable reference
Const      # Literal constant
Prim       # Primitive operation
Apply      # Function application
Lambda     # Lambda abstraction
Let        # Let binding
IfThenElse # Conditional
```

**Type System:**
```python
PrimitiveType  # Grid, Object, Int, Bool, etc.
ListType       # List[T]
FunctionType   # (T1, T2, ...) -> R
```

**Features:**
- Serialization to/from JSON
- Program size and depth metrics
- Substitution and variable extraction
- Pretty printing

#### `dsl/interpreter.py`

**Interpreter Components:**
```python
class Environment:
    """Variable bindings with lexical scoping."""

class Closure:
    """Function values with captured environment."""

class Interpreter:
    """Main evaluation engine."""
```

**Features:**
- Call-by-value semantics
- Lexical scoping with closures
- Error handling with informative messages
- Step counting for termination
- Debug mode for tracing

### 3. `synth` - Program Synthesis

Multiple synthesis strategies for finding programs.

#### `synth/enumerator.py`

**Three Synthesizers:**

1. **EnumerativeSynthesizer**
   - Bottom-up enumeration by depth
   - Type-directed generation
   - Observational equivalence pruning
   - Configurable limits (depth, size, count)

2. **StochasticSynthesizer**
   - Monte Carlo sampling
   - Weighted primitive selection
   - Temperature control
   - Partial credit scoring

3. **GeneticSynthesizer**
   - Population-based evolution
   - Tournament selection
   - Subtree crossover and mutation
   - Elitism for best programs

**Configuration:**
```python
@dataclass
class SynthesisConfig:
    max_depth: int
    max_size: int
    max_programs: int
    timeout_seconds: float
    primitives: List[str]
```

**Search Space:**
- Programs of depth d have ~O(|P|^d) possibilities
- Pruning reduces by 10-100x
- Parallel evaluation possible

### 4. `solvers` - Specialized Solvers

Pattern-specific solvers for common ARC motifs.

#### `solvers/symmetry.py`
- Detects symmetry transformations
- Tests all reflection and rotation types
- Returns transformation if consistent

#### `solvers/tiling.py`
- Identifies tiling patterns
- Extracts base tiles
- Handles tile transformations

#### `solvers/object_tracking.py`
- Tracks object transformations
- Matches objects between grids
- Infers translation, scaling, color mapping

**Strategy:**
Each solver:
1. Analyzes training examples
2. Infers transformation pattern
3. Returns transformation function
4. Falls back to None if pattern inconsistent

### 5. `scripts` - Utilities

#### `scripts/eval.py`
Comprehensive evaluation framework:
- Load ARC dataset
- Run multiple synthesizers
- Timeout handling
- Metrics computation
- Result visualization

#### `scripts/visualize.py`
Visualization tools:
- Grid plotting with ARC color palette
- Task visualization (train + test)
- Solution comparison
- Difference highlighting
- Animation creation

## Data Flow

### Synthesis Pipeline

```
Input Examples
     │
     ▼
┌────────────────┐
│ Pattern Solvers│ ─────▶ Quick pattern match
└────────────────┘
     │ (if no match)
     ▼
┌────────────────┐
│  Synthesizers  │
└────────────────┘
     │
     ├─▶ Enumerative ─────▶ Exhaustive search
     ├─▶ Stochastic  ─────▶ Random sampling
     └─▶ Genetic     ─────▶ Evolutionary search
     │
     ▼
Program Candidates
     │
     ▼
┌────────────────┐
│   Evaluation   │ ─────▶ Test on examples
└────────────────┘
     │
     ▼
Solutions
```

### Interpretation Pipeline

```
Program (AST)
     │
     ▼
┌────────────────┐
│  Environment   │ ─────▶ input = test_grid
└────────────────┘
     │
     ▼
┌────────────────┐
│  Interpreter   │
└────────────────┘
     │
     ├─▶ Eval Prim    ─────▶ Lookup & apply
     ├─▶ Eval Apply   ─────▶ Evaluate func & args
     ├─▶ Eval Lambda  ─────▶ Create closure
     └─▶ Eval Let     ─────▶ Extend environment
     │
     ▼
Output Grid
```

## Design Principles

### 1. Modularity
- Each module has clear boundaries
- Minimal coupling between layers
- Easy to extend with new primitives/solvers

### 2. Type Safety
- Comprehensive type hints throughout
- Type signatures in DSL
- Runtime type checking in interpreter

### 3. Performance
- Numpy for grid operations (vectorized)
- Caching for program evaluation
- Early pruning in synthesis
- Parallel evaluation support

### 4. Extensibility
- Registry pattern for primitives
- Plugin architecture for solvers
- Configurable synthesis strategies
- Custom primitive definition

### 5. Research-Oriented
- Clean abstractions for experimentation
- Comprehensive metrics
- Serializable programs
- Reproducible results

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Grid transform | O(h×w) | Linear in grid size |
| Connected components | O(h×w) | Flood fill |
| Symmetry detection | O(h×w) | Constant number of checks |
| Enumerative synthesis | O(|P|^d) | Exponential in depth |
| Program evaluation | O(size) | Linear in AST size |

### Space Complexity

| Data Structure | Space | Notes |
|----------------|-------|-------|
| Grid | O(h×w) | Numpy array |
| Object | O(area) | Set of positions |
| Program | O(size) | AST nodes |
| Environment | O(depth) | Call stack depth |

## Testing Strategy

### Unit Tests
- `arc_core/tests/test_grid.py`: Grid operations
- Each module has corresponding tests
- Property-based testing for invariants

### Integration Tests
- End-to-end synthesis on simple tasks
- Interpreter on known programs
- Solver evaluation

### Benchmarks
- Performance on ARC eval set
- Synthesis time vs accuracy
- Memory usage profiling

## Extension Points

### Adding New Primitives

```python
from dsl.primitives import Primitive

MY_OP = Primitive(
    name="my_op",
    impl=lambda grid: transform(grid),
    type_sig="Grid -> Grid",
    description="My custom operation"
)
```

### Adding New Solvers

```python
def solve_my_pattern(examples):
    # Analyze pattern
    # Return transformation function or None
    pass
```

### Adding New Synthesis Strategies

```python
class MySynthesizer:
    def synthesize(self, examples, max_results):
        # Custom synthesis logic
        return programs
```

## Future Directions

1. **Neural-Guided Synthesis**
   - Learn to predict primitive sequences
   - Use neural networks to score programs

2. **Abstraction Learning**
   - Automatically discover new primitives
   - Build hierarchical abstractions

3. **Meta-Learning**
   - Transfer knowledge across tasks
   - Few-shot task adaptation

4. **Parallel Synthesis**
   - Distributed evaluation
   - GPU acceleration for neural components

5. **Formal Verification**
   - Prove program correctness
   - Generate certified solvers
