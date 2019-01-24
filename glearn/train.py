from glearn.policies import load_policy
from glearn.trainers import load_trainer
from glearn.utils.config import load_config
from glearn.utils.log import log_error


def train(config_path, version=None, render=False, debug=False, profile=False):
    # load config
    config = load_config(config_path, version=version, render=render, debug=debug)

    # run each evaluation
    for evaluation_config in config:
        try:
            # create policy
            policy = load_policy(evaluation_config)

            # create trainer
            trainer = load_trainer(evaluation_config, policy)

            # train policy
            trainer.train(render=render, profile=profile)
        except Exception as e:
            log_error(f"Evaluation failed: {e}")
