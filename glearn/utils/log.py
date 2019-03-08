from inspect import getframeinfo, stack
from glearn.utils.printing import colorize


def log(message, color=None, bold=False, highlight=False):
    if color is not None:
        message = colorize(message, color, bold=bold, highlight=highlight)
    print(message)


def log_warning(message, once=False):
    if once and _log_call(message):
        return

    log(message, color="yellow")


def log_error(message, once=False):
    if once and _log_call(message):
        return

    log(message, color="red", bold=True)


class Loggable(object):
    @property
    def debugging(self):
        return False

    def debug(self, message, color=None, bold=False, highlight=False):
        if self.debugging:
            log(message, color=color, bold=bold, highlight=highlight)

    def log(self, message, color=None, bold=False, highlight=False):
        log(message, color=color, bold=bold, highlight=highlight)

    def warning(self, message, once=False):
        if once and _log_call(message):
            return

        log_warning(message)

    def error(self, message, once=False):
        if once and _log_call(message):
            return

        log_error(message)


_log_calls = {}


def _log_call(message):
    # log callstack
    caller = getframeinfo(stack()[2][0])
    key = f"{caller.filename}:{caller.lineno}"
    if _log_calls.get(key, None) == message:
        return True
    _log_calls[key] = message
    return False
