from glearn.utils.subprocess_utils import shell_call
from glearn.utils.log import log, log_warning
import psutil


class MemoryStatus(object):
    def __init__(self, total=None, used=None, available=None, percent=None):
        self.total = total
        self.used = used
        self.available = available
        self.percent = percent

    def __str__(self):
        return f"Total: {self.total}, Used: {self.used}, Available: {self.available}"


def get_virtual_memory():
    mem = psutil.virtual_memory()
    return MemoryStatus(
        total=_format_bytes(mem.total),
        used=_format_bytes(mem.used),
        available=_format_bytes(mem.available),
        percent=_format_percent(mem.percent),
    )


def print_virtual_memory(label=None):
    mem = get_virtual_memory()
    label = f"Virtual ({label})" if label else "Virtual"
    _print_memory(mem, label=label)


def get_swap_memory():
    mem = psutil.swap_memory()
    return MemoryStatus(
        total=_format_bytes(mem.total),
        used=_format_bytes(mem.used),
        available=_format_bytes(mem.free),
        percent=_format_percent(mem.percent),
    )


def print_swap_memory(label=None):
    mem = get_swap_memory()
    label = f"Swap ({label})" if label else "Swap"
    _print_memory(mem, label=label)


def get_gpu_memory():
    mem = MemoryStatus()
    try:
        result = shell_call(['nvidia-smi', '-q', '-d', 'MEMORY'], response_type="text")
        lines = result.split("FB Memory Usage")[1].split("\n")[1:4]
        for line in lines:
            parts = line.split(":")
            label = parts[0].strip().lower()
            value = parts[1].strip()
            if label == "total":
                mem.total = value
            elif label == "used":
                mem.used = value
            elif label == "free":
                mem.available = value
    except Exception as e:
        log_warning(f"Failed to collect GPU information: {e}")
    return mem


def print_gpu_memory(label=None):
    mem = get_gpu_memory()
    label = f"GPU ({label})" if label else "GPU"
    _print_memory(mem, label=label)


def _format_bytes(value):
    # TODO - detect smaller units
    return f"{value >> 20} MiB"


def _format_percent(value):
    return f"{value}%"


def _print_memory(mem, label):
    log(f"Memory | {label} | {mem}", "magenta", bold=True)
