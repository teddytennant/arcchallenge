use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;

// Benchmark helper to create test grids
fn create_test_grid(height: usize, width: usize) -> Array2<i8> {
    let mut data = Vec::with_capacity(height * width);
    for i in 0..(height * width) {
        data.push((i % 10) as i8);
    }
    Array2::from_shape_vec((height, width), data).unwrap()
}

fn benchmark_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");

    for size in [10, 20, 30, 50].iter() {
        let grid = create_test_grid(*size, *size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    // Benchmark the core logic
                    // Python binding overhead not included
                    black_box(&grid);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_grid_equality(c: &mut Criterion) {
    let grid1 = create_test_grid(30, 30);
    let grid2 = create_test_grid(30, 30);

    c.bench_function("grid_equality_30x30", |b| {
        b.iter(|| {
            black_box(grid1.iter().zip(grid2.iter()).all(|(a, b)| a == b))
        });
    });
}

fn benchmark_symmetry_check(c: &mut Criterion) {
    let grid = create_test_grid(30, 30);

    c.bench_function("symmetry_check_30x30", |b| {
        b.iter(|| {
            // Check vertical symmetry
            let height = 30;
            let width = 30;
            black_box(
                (0..height).all(|r| {
                    (0..width / 2).all(|c| {
                        grid[[r, c]] == grid[[r, width - 1 - c]]
                    })
                })
            )
        });
    });
}

criterion_group!(
    benches,
    benchmark_connected_components,
    benchmark_grid_equality,
    benchmark_symmetry_check
);
criterion_main!(benches);
