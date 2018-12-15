from glearn.utils.printing import colorize


def log(self, *args):
    print(*args)


def log_warning(self, message):
    log(colorize(message, "yellow"))


def log_error(self, message):
    log(colorize(message, "red"))


class Loggable(object):
    def log(self, *args):
        print(*args)

    def warning(self, message):
        log_warning(message)

    def error(self, message):
        log_error(message)
