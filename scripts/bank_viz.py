#!/usr/bin/env python3
"""
Shared Memory Bank Conflict Visualizer

Visualize how elements map to shared memory banks (32 banks, 4 bytes each).
Helps identify bank conflicts from stride patterns.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_bank(index: int, dtype_bytes: int) -> int:
    """Calculate bank number for element at index."""
    byte_offset = index * dtype_bytes
    return (byte_offset // 4) % 32


def get_dtype_bytes(dtype: str) -> int:
    """Get byte size for dtype."""
    sizes = {
        "half": 2,
        "float": 4,
        "double": 8,
        "int": 4,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
    }
    return sizes.get(dtype.lower(), 4)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize shared memory bank assignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bank_viz.py --rows 4 --cols 32 --dtype half --pad 0
  python bank_viz.py --rows 4 --cols 128 --dtype half --pad 8
  python bank_viz.py --rows 4 --cols 16 --dtype float --pad 0
  python bank_viz.py --rows 4 --cols 128 --dtype half --pad 0 -o conflict.png
        """,
    )
    parser.add_argument("--rows", type=int, default=4, help="Number of rows")
    parser.add_argument("--cols", type=int, default=32, help="Number of columns")
    parser.add_argument("--dtype", type=str, default="half", 
                        help="Data type: half, float, double, int, int8, int16, int32, int64")
    parser.add_argument("--pad", type=int, default=0, help="Padding elements per row")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (e.g., output.png). If not specified, displays interactively.")
    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    pad = args.pad
    dtype_bytes = get_dtype_bytes(args.dtype)
    stride = cols + pad

    # Calculate bank for each element
    banks = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            idx = r * stride + c
            banks[r, c] = get_bank(idx, dtype_bytes)

    # Calculate metrics
    stride_bytes = stride * dtype_bytes
    bank_shift = (stride_bytes // 4) % 32

    # Determine conflict status
    if bank_shift == 0:
        status = "CONFLICT! All rows hit same banks"
        status_color = "red"
    elif bank_shift < 8:
        status = f"Partial conflict (shift < 8)"
        status_color = "orange"
    else:
        status = "Good (no conflicts)"
        status_color = "green"

    # Create figure
    fig_width = max(8, cols * 0.3)
    fig_height = max(3, rows * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create colormap for 32 banks
    cmap = plt.colormaps.get_cmap('tab20').resampled(32)

    # Plot heatmap
    im = ax.imshow(banks, cmap=cmap, vmin=0, vmax=31, aspect='equal')

    # Add text annotations (bank numbers) if not too many cells
    if rows * cols <= 512:
        for r in range(rows):
            for c in range(cols):
                bank = banks[r, c]
                # Choose text color based on background brightness
                text_color = 'white' if bank % 2 == 0 else 'black'
                ax.text(c, r, str(bank), ha='center', va='center', 
                       fontsize=max(6, min(10, 200 // cols)), color=text_color)

    # Labels
    ax.set_xticks(range(0, cols, max(1, cols // 16)))
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f'row {r}' for r in range(rows)])
    ax.set_xlabel('Column')

    # Title with info
    title = f"Shared Memory Bank Layout: {rows}×{cols} ({args.dtype})\n"
    title += f"Stride: {stride} elements ({stride_bytes} bytes) | "
    title += f"Bank shift/row: {bank_shift}"
    ax.set_title(title, fontsize=10)

    # Add status text
    fig.text(0.5, 0.02, f"Status: {status}", ha='center', fontsize=12, 
             color=status_color, weight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Bank ID (0-31)')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
