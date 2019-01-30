import traceback
from glearn.policies import load_policy
from glearn.trainers import load_trainer
from glearn.utils.config import load_config
from glearn.utils.log import log_error


def train(config_path, version=None, render=False, debug=False, profile=False, training=True):
    # load config
    config = load_config(config_path, version=version, render=render, debug=debug,
                         training=training)

    # run each evaluation
    for evaluation_config in config:
        policy = None
        trainer = None

        try:
            # create policy
            policy = load_policy(evaluation_config)

            # create trainer
            trainer = load_trainer(evaluation_config, policy)

            # start session  (TODO - refactor this to be cleaner)
            config.start_session()
            policy.start_session()

            # start evaluation
            trainer.start(render=render, profile=profile)
        except Exception as e:
            log_error(f"Evaluation failed: {e}")
            traceback.print_exc()
        finally:
            # cleanup session after evaluation
            if policy:
                policy.stop_session()
            config.close_session()
