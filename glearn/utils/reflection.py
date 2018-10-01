import pkg_resources


def get_function(identifier, **default_kwargs):
    # parse identifier as callable or string or dict
    if callable(identifier):
        name = None
    elif isinstance(identifier, str):
        name = identifier
    elif isinstance(identifier, dict):
        name = identifier['name']
        args = identifier.get('args', None)
        if args is not None:
            if default_kwargs is not None:
                args.update(default_kwargs)
            default_kwargs = args
    else:
        raise ValueError(f"Unkown class or function identifier: {identifier}")

    if name is None:
        # just call identifier
        result = identifier
    else:
        # get entry point, and if necessary wrap with args
        entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
        result = entry_point.load(False)

    # default arguments wrapper
    if len(default_kwargs) > 0:
        def result_wrapper(*args, **kwargs):
            actual_kwargs = default_kwargs.copy()
            actual_kwargs.update(kwargs)
            return result(*args, **actual_kwargs)
        return result_wrapper
    else:
        return result


def get_class(identifier, **default_kwargs):
    return get_function(identifier, **default_kwargs)
