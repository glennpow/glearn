import gym
from glearn.utils.reflection import get_class


def load_env(definition):
    if isinstance(definition, dict) or ":" in definition:
        # use EntryPoint to get env
        EnvClass = get_class(definition)
        env = EnvClass()

        if isinstance(definition, dict):
            env.name = definition['name']
        else:
            env.name = definition
        env.name = env.name.split(":")[-1]
    elif "-v" in definition:
        # use gym to get env
        env = gym.make(definition)
        env.name = definition
    else:
        raise Exception(f"Unrecognizable environment identifier: {definition}")
    env.definition = definition
    return env
