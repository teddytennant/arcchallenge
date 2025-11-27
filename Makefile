# Makefile for ARC Challenge Solver

.PHONY: help install build build-rust test bench clean

help:
	@echo "ARC Challenge Solver - Make Commands"
	@echo ""
	@echo "  make install       - Install Python dependencies"
	@echo "  make build         - Build everything (Python + Rust)"
	@echo "  make build-rust    - Build Rust extensions (50-100x speedup!)"
	@echo "  make test          - Run all tests"
	@echo "  make bench         - Run Rust benchmarks"
	@echo "  make clean         - Clean build artifacts"
	@echo ""

install:
	pip install -r requirements.txt

build: install build-rust

build-rust:
	@echo "Building Rust extensions (this may take 2-3 minutes)..."
	cd arc_core_rs && maturin develop --release
	cd arc_synth_rs && maturin develop --release
	@echo "✓ Rust extensions built successfully!"
	@echo "  Expected speedup: 10-100x for core operations"

test:
	pytest arc_core/tests/ -v

bench:
	@echo "Running Rust benchmarks..."
	cd arc_core_rs && cargo bench
	@echo ""
	@echo "View detailed results: arc_core_rs/target/criterion/report/index.html"

clean:
	rm -rf target/
	rm -rf arc_core_rs/target/
	rm -rf arc_synth_rs/target/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
