from collections import abc
import numpy as np


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def _format_type(value):
    typename = type(value).__name__
    if isinstance(value, str):
        pass
    elif np.isscalar(value):
        value = f"{value:.5g}"
    else:
        if isinstance(value, abc.Iterable):
            typename = f"{typename}{np.shape(value)}"
        value = f"{value}".split('\n')[0]
    return typename, value


def print_tabular(values, color=None, bold=False, show_type=True):
    # Create strings for printing
    formatted = []
    for (key, val) in values.items():
        ftype, fval = _format_type(val)
        if show_type:
            formatted.append([key, fval, ftype])
        else:
            formatted.append([key, fval])

    # Find max widths
    padding = 2
    lcols = [[len(v) for v in f] for f in np.transpose(formatted)]
    widths = np.array([np.max(f) for f in lcols]) + (2 * padding)

    # Write out the data
    dashes = '-' * np.sum(widths)
    lines = [dashes]
    for f in formatted:
        # import ipdb; ipdb.set_trace()  # HACK DEBUGGING !!!
        cols = [(" " * padding + f[i]).ljust(widths[i]) for i in range(len(f))]
        lines.append("|".join(cols))
    lines.append(dashes)
    message = '\n'.join(lines)
    if color is not None:
        message = colorize(message, color, bold=bold)
    print(message)
