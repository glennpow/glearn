import numpy as np
from collections import OrderedDict


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


def format_value(value):
    if np.isscalar(value) and not isinstance(value, str):
        return f"{value:.5g}"
    else:
        return f"{value}".split('\n')[0]


def print_tabular(values, color=None, bold=False):
    # Create strings for printing
    key2str = OrderedDict()
    for (key, val) in values.items():
        formatted = format_value(val)
        key2str[key] = formatted

    # Find max widths
    keywidth = max(map(len, key2str.keys()))
    valwidth = max(map(len, key2str.values()))

    # Write out the data
    dashes = '-' * (keywidth + valwidth + 7)
    lines = [dashes]
    for (key, val) in key2str.items():
        lines.append('| %s%s | %s%s |' % (
            key,
            ' ' * (keywidth - len(key)),
            val,
            ' ' * (valwidth - len(val)),
        ))
    lines.append(dashes)
    message = '\n' + '\n'.join(lines) + '\n'
    if color is not None:
        message = colorize(message, color, bold=bold)
    print(message)
