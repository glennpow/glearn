from glearn.utils.subprocess_utils import shell_call
from glearn.utils.log import log, log_warning
import psutil


def get_virtual_memory():
    mem = psutil.virtual_memory()
    return {
        "total": _format_memory(mem.total),
        "used": _format_memory(mem.used),
        "free": _format_memory(mem.free),
        "available": _format_memory(mem.available),
    }


def print_virtual_memory(label=None):
    mem = get_virtual_memory()
    label = f"Virtual ({label})" if label else "Virtual"
    _print_memory(mem, label=label)


def get_gpu_memory():
    mem = {}
    try:
        result = shell_call(['nvidia-smi', '-q', '-d', 'MEMORY'], response_type="text")
        lines = result.split("FB Memory Usage")[1].split("\n")[1:4]
        for line in lines:
            parts = line.split(":")
            label = parts[0].strip()
            mem[label.lower()] = parts[1].strip()
    except Exception as e:
        log_warning(f"Failed to collect GPU information: {e}")
    return mem


def print_gpu_memory(label=None):
    mem = get_gpu_memory()
    label = f"GPU ({label})" if label else "GPU"
    _print_memory(mem, label=label)


def _format_memory(value):
    # TODO - detect smaller units
    return f"{value >> 20} MiB"


def _print_memory(mem, label=None):
    label_s = f"{label} | " if label else ""
    mem_s = ", ".join([f"{k}: {v}" for k, v in mem.items()])
    log(f"Memory | {label_s}{mem_s}", "magenta", bold=True)
