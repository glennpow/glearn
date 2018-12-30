def get_device(cpu=None, gpu=None):
    device = None
    if cpu is not None and cpu is not False:
        if cpu is not True and isinstance(cpu, int):
            device = f"/cpu:{cpu}"
        else:
            device = "/cpu:0"
    elif gpu is not None and gpu is not False:
        if gpu is not True and isinstance(gpu, int):
            device = f"/device:GPU:{gpu}"
        else:
            device = "/device:GPU:0"
    return device
