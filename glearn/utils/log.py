from glearn.utils.printing import colorize


def log(message, color=None, bold=False, highlight=False):
    if color is not None:
        message = colorize(message, color, bold=bold, highlight=highlight)
    print(message)


def log_warning(message):
    log(message, color="yellow")


def log_error(message):
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

    def warning(self, message):
        log_warning(message)

    def error(self, message):
        log_error(message)
