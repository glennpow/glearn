from collections import abc, OrderedDict
import numpy as np


COLOR_NUMBERS = dict(
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
MAX_TABULAR_WIDTH = 120


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = COLOR_NUMBERS[color]
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
    if len(value) > MAX_TABULAR_WIDTH:
        value = value[:MAX_TABULAR_WIDTH]
    return typename, value


def print_tabular(values, grouped=False, color=None, bold=False, show_type=True, padding=2):
    # Create strings for printing
    if not grouped:
        values = {None: values}
    formatted = OrderedDict()
    for header, group in values.items():
        group_format = []
        formatted[header] = group_format
        for (key, val) in group.items():
            ftype, fval = _format_type(val)
            if show_type:
                group_format.append([key, fval, ftype])
            else:
                group_format.append([key, fval])

    # Find max widths
    lcols = {header: [[len(v) for v in f] for f in np.transpose(group)]
             for header, group in formatted.items()}
    widths = {header: np.array([np.max(f) for f in group]) + (2 * padding)
              for header, group in lcols.items()}
    table_width = max([np.sum(w) for _, w in widths.items()])

    # header and table widths
    header_widths = {}
    for header, group in formatted.items():
        if header is not None:
            header_width = len(header) + (2 * padding)
            header_widths[header] = header_width
            table_width = max(header_width, table_width)

    # Write out the data
    equals = '=' * table_width
    dashes = '-' * table_width
    dotted = ('- ' * (table_width // 2 + 1))[:table_width]
    lines = []
    for header, group in formatted.items():
        lines.append(equals if len(lines) == 0 else dashes)
        if header is not None:
            lines += [" " * padding + header, dotted]
        group_widths = widths[header]
        for f in group:
            cols = [(" " * padding + f[i]).ljust(group_widths[i]) for i in range(len(f))]
            lines.append("|".join(cols))
    lines.append(equals)
    message = '\n'.join(lines)
    if color is not None:
        message = colorize(message, color, bold=bold)
    print(message)
