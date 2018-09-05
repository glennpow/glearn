import json
import logging
from subprocess import check_call, check_output, Popen, PIPE
from inspect import signature
from utils.printing import colorize


logger = logging.getLogger(__name__)


def shell_call(command, verbose=False, ignore_exceptions=False, response_type=None, env=None,
               **kwargs):
    """
    Wrapper around the system calls.

    Parameters
    ----------
    command : list(str) or str
        The command call, which will be passed to subprocess
    verbose : bool or int
        Whether or not to print out the command and its result.  (Default: False)
        If an int is provided, then this will be the string limit of the printed result.
    ignore_exceptions : bool
        Whether or not to ignore_exceptions all exceptions from this call.  (Default: False)
    response_type : string
        Determines how to handle stdout response from process.
        None : Don't capture stdout  (Default)
        "text" : Capture and return raw stdout.
        "json" : Capture stdout and return parses json object.
    env : Mapping
        Environment variables map.  (Default: None)
    **kwargs
        Any additional args will be passed to subprocess call

    Returns
    -------
    str
        Command response.  Can be None if no response received.
    """
    if isinstance(command, list):
        command = [str(token) for token in command]
        command_str = ' '.join(command)

        if "shell" in kwargs and kwargs["shell"]:
            command = command_str
    elif isinstance(command, str):
        command_str = command
    try:
        if verbose:
            _log(f"Calling command: {command_str}", "green", bold=True)

        # call system command
        if response_type is not None:
            output = check_output(command, universal_newlines=True, env=env, **kwargs).strip()
            if len(output) > 0:
                _log_output(output, verbose)

                if response_type == "json":
                    # parse json result
                    return json.loads(output)
                return output
        else:
            check_call(command, env=env, **kwargs)
    except Exception as e:
        _log(f"Exception encountered during system call ({command_str}):  {e}", "red", bold=True)
        if not ignore_exceptions:
            raise e
    return None


def shell_calls(commands, verbose=False, ignore_exceptions=False, response_type=None, env=None,
                **kwargs):
    """
    Run multiple system commands serially.  They are concatenated with '&&'

    Parameters
    ----------
    commands : list(list(str)) or list(str)
        The command calls, which will be passed to subprocess
    verbose : bool or int
        Whether or not to print out the command and its result.  (Default: False)
        If an int is provided, then this will be the string limit of the printed result.
    ignore_exceptions : bool
        Whether or not to ignore_exceptions all exceptions from this call.  (Default: False)
    response_type : string
        Determines how to handle stdout response from process.
        None : Don't capture stdout  (Default)
        "text" : Capture and return raw stdout.
        "json" : Capture stdout and return parses json object.
    env : Mapping
        Environment variables map.  (Default: None)
    **kwargs
        Any additional args will be passed to subprocess calls

    Returns
    -------
    str
        Response from final command.  Can be None if no responses received.
    """
    concat = []
    for command in commands:
        if len(concat) > 0:
            concat += ["&&"]
        if isinstance(command, list):
            concat += command
        elif isinstance(command, str):
            concat.append(command)
    return shell_call(concat, verbose=verbose, ignore_exceptions=ignore_exceptions,
                      response_type=response_type, env=env, **kwargs)


def parallel_shell_calls(commands, verbose=False, ignore_exceptions=False,
                         response_type=None, env=None, **kwargs):
    """
    Run multiple system commands in parallel

    Parameters
    ----------
    command : list(list(str)) or list(str)
        The command calls, which will be parallel-passed to Popen
    verbose : bool or int
        Whether or not to print out the command and its result.  (Default: False)
        If an int is provided, then this will be the string limit of the printed result.
    ignore_exceptions : bool
        Whether or not to ignore_exceptions all exceptions from this call.  (Default: False)
    response_type : string
        Determines how to handle stdout response from process.
        None : Don't capture stdout  (Default)
        "text" : Capture and return raw stdout.
        "json" : Capture stdout and return parses json object.
    env : Mapping
        Environment variables map.  (Default: None)
    **kwargs
        Any additional args will be passed to Popen()

    Returns
    -------
    list(str)
        Command responses.  Can be None if no responses received.
    """
    calls = []
    for command in commands:
        if isinstance(command, list):
            command = [str(token) for token in command]
            command_str = ' '.join(command)
        elif isinstance(command, str):
            command_str = command
        try:
            if verbose:
                _log(f"Calling command: {command_str}", "green", bold=True)

            process = Popen(command, universal_newlines=True, stdout=PIPE, env=env, **kwargs)
            calls.append([command_str, process])
        except Exception as e:
            _log(f"Exception during system call ({command_str}):  {e}", "red", bold=True)
            if not ignore_exceptions:
                raise e
    responses = [] if response_type is not None else None
    for command_str, process in calls:
        try:
            output = process.communicate()[0]
            if len(output) > 0:
                _log_output(output, verbose)

                if responses is not None:
                    if response_type == "json":
                        # parse json result
                        responses.append(json.loads(output))
                    else:
                        responses.append(output)
        except Exception as e:
            _log(f"Exception during system call ({command_str}):  {e}", "red", bold=True)
            if not ignore_exceptions:
                raise e
    return responses


def parse_command_args(parser, commands, performer=None):
    """
    Adds subparser arguments for a list of commands for a CLI call.

    Parameters
    ----------
    parser : ArgumentParser
        The base argparse object created for the CLI call.
    commands : list(types.FunctionType)
        A list of functions to be runnable commands.
        The functions can either take a single "args" parameter,
        which will be the parsed ArgumentParser object,
        or it can take any other keyword parameters, and they will be retrieved via
        calling **vars(args).
        The function's name will be expected as the CLI command argument.
        If the function has a custom property "usage", which points to another function,
        then this function will be called to add the various subparser arguments.
    performer : function(name=str, args=Arguments, perform=function())
        An optional function that can wrap the call of the specified command.  (Default: None)
        The performing command 'name' and 'args' are supplied.
        This performer function should call the 'perform' argument when execution is desired.

    Examples
    --------
        def foo_usage(parser):
            parser.add_argument("--bar", action="store_true")

        def foo(args):
            print(f"User: {args.user}, Bar: {args.bar}")

        foo.usage = foo_usage

        def far_usage(parser):
            parser.add_argument("--boo", action="store_true")

        def far(user=None, boo=False):
            print(f"User: {user}, Boo: {boo}")

        far.usage = far_usage

        def performer(name, args, perform):
            print(f"We're about to perform command: '{name}'' as user: '{args.user}'")
            perform()
            print("Done performing command!")

        parser = argparse.ArgumentParser()
        parser.add_argument("--user", type=str, default="root")
        parse_command_args(parser, [foo, far], performer=performer)

        # $ python main.py --user=john far --boo
    """
    subparsers = parser.add_subparsers()

    for i in range(len(commands)):
        command = commands[i]

        def parse_command(subparser, command):
            sig = signature(command)
            if len(sig.parameters) == 1 and "args" in sig.parameters:
                command_func = command
            else:
                def explode_command_func(args):
                    kwargs = vars(args)
                    kwargs.pop("func", None)
                    command(**kwargs)
                command_func = explode_command_func
            command_func = _wrap_performer(command, command_func, performer)
            subparser.__performer__ = performer
            if hasattr(parser, "__performer__"):
                command_func = _wrap_performer(command, command_func, parser.__performer__)
            subparser.set_defaults(func=command_func)
            subparser.__inner__ = True

            if hasattr(command, "usage"):
                command.usage(subparser)

        command_name = command.__name__
        subparser = subparsers.add_parser(command_name)
        parse_command(subparser, command)

    if not hasattr(parser, "__inner__"):
        # run desired command, unless it's an inner parser
        args, unparsed = parser.parse_known_args()
        args.__unparsed__ = unparsed
        args.func(args)


def _wrap_performer(command, command_func, performer):
    if performer is not None:
        perform_command_func = command_func

        def performer_command_func(args):
            def perform():
                perform_command_func(args)

            performer(command.__name__, args, perform)
        command_func = performer_command_func
    return command_func


def _log(message, color=None, bold=False):
    if color is not None:
        message = colorize(message, color, bold=bold)
    logger.info(message)


def _log_output(output, verbose):
    if verbose:
        if isinstance(verbose, int) and len(output) > verbose:
            output = f"{output[:verbose]}..."
        _log(output, "green")
