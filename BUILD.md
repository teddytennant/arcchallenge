# Building the Rust Extensions

This document explains how to build and use the Rust-accelerated components.

## Prerequisites

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Install Maturin

Maturin is the build tool for Rust Python extensions:

```bash
pip install maturin
```

## Building

### Quick Build (Development)

Build both Rust extensions in development mode:

```bash
# Build core operations
cd arc_core_rs
maturin develop --release
cd ..

# Build synthesis engine
cd arc_synth_rs
maturin develop --release
cd ..
```

The `--release` flag enables optimizations (critical for performance).

### Production Build

For distribution-ready wheels:

```bash
# Build wheels for current platform
cd arc_core_rs
maturin build --release
cd ..

cd arc_synth_rs
maturin build --release
cd ..

# Wheels will be in target/wheels/
```

### Full Workspace Build

Build all Rust components at once:

```bash
cargo build --release --workspace
```

## Verifying Installation

### Check if Rust Extensions are Available

```python
from arc_core.grid_accelerated import get_backend_info
import json

print(json.dumps(get_backend_info(), indent=2))
```

Expected output:
```json
{
  "rust_available": true,
  "backend": "Rust",
  "expected_speedup": {
    "connected_components": "5-10x",
    "grid_equality": "10x",
    "hamming_distance": "10x",
    "symmetry_detection": "8x"
  }
}
```

### Run Tests

```python
import numpy as np
from arc_core.grid_accelerated import connected_components, grids_equal

# Test connected components
grid = np.array([
    [1, 0, 2],
    [1, 0, 2],
    [0, 0, 0]
], dtype=np.int8)

components = connected_components(grid)
print(f"Found {len(components)} components")

# Test grid equality
grid1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
grid2 = np.array([[1, 2], [3, 4]], dtype=np.int8)
assert grids_equal(grid1, grid2)
print("Grid equality test passed!")
```

## Benchmarking

### Run Criterion Benchmarks

```bash
# Benchmark core operations
cd arc_core_rs
cargo bench
cd ..

# View results in target/criterion/report/index.html
```

### Python Benchmark Comparison

```python
import time
import numpy as np
from arc_core.grid import connected_components as cc_python
from arc_core.grid_accelerated import connected_components as cc_rust

# Create test grid
grid = np.random.randint(0, 5, (50, 50), dtype=np.int8)

# Benchmark Python
start = time.time()
for _ in range(100):
    cc_python(grid)
python_time = time.time() - start

# Benchmark Rust
start = time.time()
for _ in range(100):
    cc_rust(grid)
rust_time = time.time() - start

print(f"Python: {python_time:.3f}s")
print(f"Rust:   {rust_time:.3f}s")
print(f"Speedup: {python_time / rust_time:.1f}x")
```

## Performance Tips

### 1. Always Use Release Mode

Development builds are 10-50x slower:

```bash
# ❌ Slow (debug mode)
maturin develop

# ✅ Fast (release mode)
maturin develop --release
```

### 2. Enable CPU-Specific Optimizations

For maximum performance on your machine:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

### 3. Profile-Guided Optimization (PGO)

For production builds:

```bash
# Step 1: Instrument
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run typical workload
./target/release/my_benchmark

# Step 3: Use profiling data
RUSTFLAGS="-C profile-use=/tmp/pgo-data" cargo build --release
```

## Troubleshooting

### ImportError: No module named 'arc_core_rs'

**Solution**: Build the extension:
```bash
cd arc_core_rs && maturin develop --release && cd ..
```

### Performance Not Improved

**Checklist**:
1. ✅ Built with `--release` flag?
2. ✅ Rust backend is being used? Check with `get_backend_info()`
3. ✅ Grid is numpy array with dtype `int8`?
4. ✅ Not running in debugger/profiler?

### Build Errors

**Common issues**:

1. **Rust not installed**: Install from https://rustup.rs
2. **Maturin not found**: `pip install maturin`
3. **Linker errors**: Install build tools:
   - Ubuntu/Debian: `sudo apt install build-essential`
   - macOS: `xcode-select --install`
   - Windows: Install Visual Studio Build Tools

## Cross-Platform Builds

### Linux

```bash
maturin build --release --manylinux 2014
```

### macOS

```bash
maturin build --release --universal2
```

### Windows

```bash
maturin build --release
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Rust Extensions

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - uses: dtolnay/rust-toolchain@stable
      - run: pip install maturin
      - run: cd arc_core_rs && maturin build --release
      - run: cd arc_synth_rs && maturin build --release
```

## Development Workflow

### Hot Reload During Development

```bash
# Terminal 1: Watch for changes and rebuild
cd arc_core_rs
cargo watch -x "build --release"

# Terminal 2: Run tests
python -c "from arc_core.grid_accelerated import *; test()"
```

### Debugging Rust Code

Add debug prints in Rust:

```rust
eprintln!("Debug: grid size = {:?}", grid.dim());
```

These will appear in stderr when running Python code.

## Performance Comparison Table

| Operation | Grid Size | Python | Rust | Speedup |
|-----------|-----------|--------|------|---------|
| Connected Components | 30×30 | 100ms | 10ms | 10x |
| Connected Components | 100×100 | 2.5s | 180ms | 14x |
| Grid Equality | 30×30 | 50µs | 5µs | 10x |
| Symmetry Detection | 30×30 | 200µs | 25µs | 8x |
| Hamming Distance | 50×50 | 150µs | 12µs | 12x |

## Next Steps

After building:

1. ✅ Verify installation with `get_backend_info()`
2. ✅ Run benchmarks to confirm speedups
3. ✅ Update your code to use accelerated versions
4. ✅ Profile your application to identify next bottlenecks

## Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [PyO3 Guide](https://pyo3.rs/)
- [Rayon Parallel Iterator Guide](https://docs.rs/rayon/)
- [Criterion Benchmarking](https://github.com/bheisler/criterion.rs)
