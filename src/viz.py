"""Figure theme and primitives for the manuscript.

One visual system across every figure. Categorical hues are assigned in a FIXED order and
never cycled; a fourth chemical class additionally carries its own marker shape so identity is
never colour-alone. Sequential scales are single-hue light to dark; the one diverging scale is
two hues around a neutral grey midpoint. No figure uses two y-scales.

The palette is validated (lightness band, chroma floor, colour-vision-deficiency separation,
normal-vision floor, contrast). Three slots clear the all-pairs gates, which is why scatter
figures use at most three hues plus the shape-encoded dual class. Three light-mode hues sit
below 3:1 contrast on the print surface, so every figure ships axis labels, a legend and a
matching CSV table rather than relying on fill contrast alone.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_3 = '#6f6e6a'
GRID = '#e4e4df'

# Fixed categorical order (validated). Slots 1-3 clear the all-pairs gates.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']

CLASS_COLOR = {0: '#2a78d6', 1: '#eb6834', 2: '#1baf7a', 3: '#4a3aa7'}
CLASS_MARKER = {0: 'o', 1: 's', 2: '^', 3: 'D'}      # secondary encoding, not colour alone
CLASS_LABEL = {0: 'Inactive', 1: 'Agonist-only', 2: 'Antagonist-only', 3: 'Dual-active'}

SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    'seq_blue', ['#eaf2fc', '#9cc2ee', '#4a8fdd', '#2a78d6', '#164a87', '#0b2744'])
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    'div_ob', ['#eb6834', '#f3a884', '#dedcd6', '#8fb6e6', '#2a78d6'])

GOOD, WARNING, SERIOUS, CRITICAL = '#1b7f4b', '#eda100', '#e07020', '#c0352f'


def apply_theme(base=9.5):
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.18,
        'font.size': base, 'font.family': 'DejaVu Sans',
        'axes.titlesize': base + 1.5, 'axes.titleweight': 'medium', 'axes.titlelocation': 'left',
        'axes.titlepad': 9, 'axes.labelsize': base, 'axes.labelcolor': INK_2,
        'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelpad': 5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-', 'grid.alpha': 1.0,
        'xtick.color': INK_3, 'ytick.color': INK_3, 'xtick.labelsize': base - 0.5,
        'ytick.labelsize': base - 0.5, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'text.color': INK,
        'legend.frameon': False, 'legend.fontsize': base - 0.5, 'legend.handlelength': 1.4,
        'legend.labelcolor': INK_2, 'legend.borderaxespad': 0.4,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round', 'lines.markersize': 4.5,
        'figure.titlesize': base + 2.5, 'figure.titleweight': 'medium',
        'errorbar.capsize': 0,
    })


def save(fig, name, out, caption=None):
    """Write PNG and SVG. Every figure also has a CSV table elsewhere (the table view)."""
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out / 'figures' / f'{name}.{ext}')
    plt.close(fig)
    if caption:
        with open(out / 'figures' / 'captions.md', 'a') as fh:
            fh.write(f'**{name}.** {caption}\n\n')
    return name


def title(ax, text, subtitle=None):
    """Title above an optional subtitle. The pad reserves the subtitle's line so they
    cannot overlap, which they do if the title keeps the default pad."""
    if subtitle:
        ax.set_title(text, color=INK, pad=24)
        ax.text(0, 1.012, subtitle, transform=ax.transAxes,
                fontsize=plt.rcParams['font.size'] - 1, color=INK_3, va='bottom')
    else:
        ax.set_title(text, color=INK, pad=9)


def rounded_bars(ax, positions, values, color=None, width=0.62, horizontal=False, radius_px=4):
    """Bars with a rounded data-end, square at the baseline, capped thickness, 2px surface gaps."""
    color = color or SERIES[0]
    out = []
    for pos, val in zip(positions, values):
        if not np.isfinite(val) or val == 0:
            continue
        pad = abs(val) * 0.0001
        if horizontal:
            box = FancyBboxPatch((0, pos - width / 2), abs(val) + pad, width,
                                 boxstyle=f'round,pad=0,rounding_size={abs(val)*0.012:.6f}',
                                 linewidth=0, facecolor=color, mutation_aspect=1)
        else:
            box = FancyBboxPatch((pos - width / 2, 0), width, abs(val) + pad,
                                 boxstyle=f'round,pad=0,rounding_size={width*0.07:.6f}',
                                 linewidth=0, facecolor=color)
        ax.add_patch(box)
        out.append(box)
    return out


def boxplot(ax, data, labels, colors=None, width=0.58, horizontal=False, showpoints=True,
            rng_seed=0):
    """Thin box plot with recessive whiskers, a surface-coloured median and jittered raw folds.

    Showing the individual fold values beside the box is deliberate: with five folds a box
    plot alone implies more distributional information than exists.
    """
    colors = colors or [SERIES[i % len(SERIES)] for i in range(len(data))]
    clean = [np.asarray([v for v in np.asarray(d, dtype=float) if np.isfinite(v)]) for d in data]
    positions = np.arange(len(clean))
    bp = ax.boxplot(clean, positions=positions, widths=width, patch_artist=True,
                    vert=not horizontal, showfliers=False, whis=(0, 100),
                    medianprops=dict(color=SURFACE, linewidth=1.8),
                    whiskerprops=dict(color=INK_3, linewidth=0.9),
                    capprops=dict(color=INK_3, linewidth=0.9),
                    boxprops=dict(linewidth=0))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(1.0)
    if showpoints:
        rng = np.random.default_rng(rng_seed)
        for i, (vals, c) in enumerate(zip(clean, colors)):
            if not len(vals):
                continue
            j = i + rng.uniform(-width * 0.22, width * 0.22, len(vals))
            xs, ys = (vals, j) if horizontal else (j, vals)
            ax.scatter(xs, ys, s=11, facecolor=INK, edgecolor=SURFACE, linewidth=1.0,
                       alpha=0.55, zorder=4)
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.grid(axis='x')
        ax.grid(axis='y', visible=False)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
    return bp


def scatter_classes(ax, xy, labels, classes=(0, 1, 2, 3), size=16, alpha=0.75, order=None):
    """Chemical-space scatter: colour plus a per-class marker shape (secondary encoding)."""
    labels = np.asarray(labels)
    order = order or [c for c in classes if c != 0] + ([0] if 0 in classes else [])
    for c in [x for x in classes if x == 0] + [x for x in order if x != 0]:
        m = labels == c
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=size if c else size * 0.55,
                   c=CLASS_COLOR[c], marker=CLASS_MARKER[c],
                   alpha=alpha if c else alpha * 0.45,
                   linewidth=0.5 if c else 0.0, edgecolor=SURFACE,
                   label=f'{CLASS_LABEL[c]} (n={int(m.sum())})', zorder=3 if c else 2)
    ax.legend(loc='best', markerscale=1.5)
    ax.grid(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(True)
    return ax


def annotate_right(ax, y, text, color=INK_2, dx=0.004):
    ax.text(1 + dx, y, text, transform=ax.get_yaxis_transform(), va='center',
            fontsize=plt.rcParams['font.size'] - 1, color=color)


def reference_line(ax, value, axis='y', label=None, color=None):
    color = color or INK_3
    fn = ax.axhline if axis == 'y' else ax.axvline
    fn(value, color=color, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    if label:
        if axis == 'y':
            ax.text(0.995, value, f' {label}', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', fontsize=plt.rcParams['font.size'] - 1.5,
                    color=color)
        else:
            ax.text(value, 0.99, f' {label}', transform=ax.get_xaxis_transform(),
                    ha='left', va='top', fontsize=plt.rcParams['font.size'] - 1.5, color=color)


def empty(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center',
            color=INK_3, fontsize=plt.rcParams['font.size'])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
