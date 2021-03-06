import os
import sys
import termios
import tty
import fcntl
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
MAX_TABULAR_WIDTH = 80


def getch(blocking=True):
    # get original terminal state
    fd = sys.stdin.fileno()
    original_termios = termios.tcgetattr(fd)
    if not blocking:
        original_fcntl = fcntl.fcntl(fd, fcntl.F_GETFL)

    try:
        # set to immediate (possibly blocking) terminal state
        tty.setraw(sys.stdin.fileno())
        if not blocking:
            fcntl.fcntl(fd, fcntl.F_SETFL, original_fcntl | os.O_NONBLOCK)

        # read single terminal character
        ch = sys.stdin.read(1)
    except IOError as e:
        # if not blocking, ignore empty input
        if blocking:
            raise e
        return 0
    finally:
        # reset to original terminal state
        if not blocking:
            fcntl.fcntl(fd, fcntl.F_SETFL, original_fcntl)
        termios.tcsetattr(fd, termios.TCSADRAIN, original_termios)
    return ch


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = COLOR_NUMBERS[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def print_update(message, color=None, bold=False, highlight=False):
    # format update info
    if isinstance(message, list):
        status = message[0]
        info = message[1:]
        status_border = "█"
        info_border = "  ░  "
        info_message = info_border.join(info)
        message = f"{status_border}  {status}  {status_border}  {info_message}"

    # print message and return to start of line
    if color is not None or bold or highlight:
        message = colorize(message, color, bold=bold, highlight=highlight)
    print(message, end="\r", flush=True)
    sys.stdout.write("\033[K")


def print_tabular(values, grouped=False, color=None, bold=False, show_type=True, padding=2,
                  squeeze=True):
    # Create strings for printing
    if not grouped:
        values = {None: values}
    formatted = OrderedDict()
    for header, group in values.items():
        group_format = []
        formatted[header] = group_format
        for (key, val) in group.items():
            ftype, fval = _format_tabular_data(val, squeeze=squeeze)
            if show_type:
                group_format.append([key, fval, ftype])
            else:
                group_format.append([key, fval])

    # Find max widths
    lcols = {header: [[len(v) for v in f] for f in np.transpose(group)]
             for header, group in formatted.items()}
    widths = {header: np.array([np.max(f) for f in group]) + (2 * padding)
              for header, group in lcols.items()}
    table_width = max([np.sum(w) + len(w) - 1 for _, w in widths.items()])

    # header and table widths
    header_widths = {}
    for header, group in formatted.items():
        if header is not None:
            header_width = len(header) + (2 * padding)
            header_widths[header] = header_width
            table_width = max(header_width, table_width)

    # Write out the data █▓▒░
    top = "┌" + ("▒" * table_width) + "┐"
    bottom = "└" + ('─' * table_width) + "┘"
    header_top = "├" + ('░' * table_width) + "┤"
    header_bottom = "├" + ('╶' * table_width) + "┤"
    lines = []
    for header, group in formatted.items():
        lines.append(top if len(lines) == 0 else header_top)
        if header is not None:
            lines += ["│" + (" " * padding + header).ljust(table_width) + "│", header_bottom]
        group_widths = widths[header]
        for f in group:
            cols = []
            n_cols = len(f)
            col_sum = 0
            for i in range(n_cols):
                col_width = group_widths[i] if i < n_cols - 1 else table_width - col_sum
                col_sum += col_width + 1
                cols.append((" " * padding + str(f[i])).ljust(col_width))
            lines.append("│" + "│".join(cols) + "│")
    lines.append(bottom)
    message = '\n'.join(lines)
    if color is not None or bold:
        message = colorize(message, color, bold=bold)
    print(message)


def _format_tabular_data(value, squeeze=False):
    typename = type(value).__name__
    if isinstance(value, str):
        pass
    elif np.isscalar(value):
        if abs(value) >= 100000:
            value = f"{value:,.0f}"
        else:
            value = f"{value:,.5g}"
    else:
        if isinstance(value, abc.Iterable):
            typename = f"{typename}{np.shape(value)}"
        value = np.array(value)
        if squeeze:
            value = np.squeeze(value)
        value = f"{value}".replace('\n', ' ')
    if len(value) > MAX_TABULAR_WIDTH:
        value = value[:MAX_TABULAR_WIDTH]
    return typename, value
