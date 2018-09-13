import pkg_resources


def get_function(identifier, default_kwargs=None):
    # parse identifier as string or dict
    if isinstance(identifier, str):
        name = identifier
    elif isinstance(identifier, dict):
        name = identifier['name']
        args = identifier.get('args', None)
        if args is not None:
            if default_kwargs is not None:
                args.update(default_kwargs)
            default_kwargs = args

    # get entry point, and if necessary wrap with args
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


def get_class(identifier, default_kwargs=None):
    return get_function(identifier, default_kwargs=default_kwargs)
