from glearn.utils.printing import colorize


def log(*args):
    print(*args)


def log_warning(message):
    log(colorize(message, "yellow"))


def log_error(message):
    log(colorize(message, "red"))


class Loggable(object):
    @property
    def debugging(self):
        return False

    def debug(self, *args):
        if self.debugging:
            log(*args)

    def log(self, *args):
        log(*args)

    def warning(self, message):
        log_warning(message)

    def error(self, message):
        log_error(message)
