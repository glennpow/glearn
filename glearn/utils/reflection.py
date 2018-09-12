import pkg_resources


def get_function(name, default_kwargs=None):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    if default_kwargs is not None and len(default_kwargs) > 0:
        def result_wrapper(*args, **kwargs):
            actual_kwargs = default_kwargs.copy()
            actual_kwargs.update(kwargs)
            return result(*args, **actual_kwargs)
        return result_wrapper
    else:
        return result


def get_class(name, default_kwargs=None):
    return get_function(name, default_kwargs=default_kwargs)
