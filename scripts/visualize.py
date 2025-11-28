"""
Visualization utilities for ARC tasks and solutions.

Provides tools to visualize grids, tasks, and program execution.
"""

from typing import List, Optional, Tuple
import numpy as np
import io
import base64

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from ..arc_core.grid import Grid
except ImportError:
    from arc_core.grid import Grid


# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: light blue
    '#870C25',  # 9: dark red/brown
]

ARC_CMAP = ListedColormap(ARC_COLORS)


def plot_grid(grid: Grid, ax=None, title: str = "", show_grid: bool = True):
    """
    Plot a single grid.

    Args:
        grid: Grid to plot
        ax: Matplotlib axis (creates new if None)
        title: Title for the plot
        show_grid: Whether to show grid lines
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    h, w = grid.shape

    # Plot grid
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')

    # Add grid lines
    if show_grid:
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
        for i in range(w + 1):
            ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)

    # Configure axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold')

    return ax


def plot_task(task_data: dict, max_examples: int = 3,
              save_path: Optional[str] = None):
    """
    Plot an ARC task with train and test examples.

    Args:
        task_data: Task dictionary with 'train' and 'test' keys
        max_examples: Maximum number of train examples to show
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    train_examples = task_data['train'][:max_examples]
    test_examples = task_data.get('test', [])

    n_train = len(train_examples)
    n_test = len(test_examples)

    # Create figure
    fig, axes = plt.subplots(
        n_train + n_test, 2,
        figsize=(8, 4 * (n_train + n_test))
    )

    if n_train + n_test == 1:
        axes = axes.reshape(1, -1)

    # Plot train examples
    for i, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        plot_grid(input_grid, ax=axes[i, 0], title=f"Train {i+1} - Input")
        plot_grid(output_grid, ax=axes[i, 1], title=f"Train {i+1} - Output")

    # Plot test examples
    for i, example in enumerate(test_examples):
        input_grid = np.array(example['input'])
        plot_grid(input_grid, ax=axes[n_train + i, 0], title=f"Test {i+1} - Input")

        # Output might not be available
        if 'output' in example:
            output_grid = np.array(example['output'])
            plot_grid(output_grid, ax=axes[n_train + i, 1], title=f"Test {i+1} - Output")
        else:
            axes[n_train + i, 1].text(0.5, 0.5, "Unknown", ha='center', va='center')
            axes[n_train + i, 1].set_xticks([])
            axes[n_train + i, 1].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_solution(input_grid: Grid, expected_output: Grid, predicted_output: Grid,
                  save_path: Optional[str] = None):
    """
    Plot input, expected output, and predicted output side by side.

    Args:
        input_grid: Input grid
        expected_output: Expected output grid
        predicted_output: Predicted output grid
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_grid(input_grid, ax=axes[0], title="Input")
    plot_grid(expected_output, ax=axes[1], title="Expected Output")
    plot_grid(predicted_output, ax=axes[2], title="Predicted Output")

    # Check if prediction is correct
    is_correct = np.array_equal(expected_output, predicted_output)
    fig.suptitle(
        f"Solution {'✓ CORRECT' if is_correct else '✗ INCORRECT'}",
        fontsize=14,
        fontweight='bold',
        color='green' if is_correct else 'red'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_grid_diff(grid1: Grid, grid2: Grid, save_path: Optional[str] = None):
    """
    Plot difference between two grids.

    Args:
        grid1: First grid
        grid2: Second grid
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if grid1.shape != grid2.shape:
        raise ValueError("Grids must have same shape")

    diff = (grid1 != grid2).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_grid(grid1, ax=axes[0], title="Grid 1")
    plot_grid(grid2, ax=axes[1], title="Grid 2")

    axes[2].imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='nearest')
    axes[2].set_title(f"Difference ({np.sum(diff)} cells)", fontsize=12, fontweight='bold')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_grid_ascii(grid: Grid, color_map: Optional[dict] = None):
    """
    Print grid as ASCII art.

    Args:
        grid: Grid to print
        color_map: Optional mapping from color values to ASCII characters
    """
    if color_map is None:
        color_map = {
            0: '.',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
        }

    for row in grid:
        print(''.join(color_map.get(val, '?') for val in row))


def create_animation(grids: List[Grid], titles: Optional[List[str]] = None,
                     save_path: str = "animation.gif", duration: int = 500):
    """
    Create an animated GIF from a sequence of grids.

    Args:
        grids: List of grids to animate
        titles: Optional titles for each frame
        save_path: Path to save GIF
        duration: Duration of each frame in milliseconds
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    try:
        from PIL import Image
        import io
    except ImportError:
        raise ImportError("Pillow is required for creating animations")

    frames = []

    for i, grid in enumerate(grids):
        fig, ax = plt.subplots(figsize=(6, 6))
        title = titles[i] if titles and i < len(titles) else f"Frame {i+1}"
        plot_grid(grid, ax=ax, title=title)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Animation saved to {save_path}")


def grid_to_base64(grid: Grid, size: Tuple[int, int] = (300, 300)) -> str:
    """
    Convert a grid to a base64-encoded PNG image for embedding in HTML.

    Args:
        grid: Grid to convert
        size: Size of the output image in pixels (width, height)

    Returns:
        Base64-encoded PNG image string
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Create figure
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    plot_grid(grid, ax=ax, title="", show_grid=True)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Encode as base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"
