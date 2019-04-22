import gym
from glearn.utils.reflection import get_class


def load_env(definition):
    if isinstance(definition, dict):
        def_name = definition["name"]
        def_args = definition.get("args", {})
    else:
        def_name = definition
        def_args = {}

    if ":" in def_name:
        # use EntryPoint to get env
        EnvClass = get_class(definition)
        env = EnvClass()

        env.name = def_name.split(":")[-1]
    elif "-v" in def_name:
        # use gym to get env
        env = gym.make(def_name, **def_args)
        env.name = def_name
    else:
        raise Exception(f"Unknown environment definition: {definition}")

    env.definition = definition
    return env
