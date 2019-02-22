import traceback
from glearn.policies import load_policy
from glearn.trainers import load_trainer
from glearn.utils.config import load_config
from glearn.utils.log import log_error, log_warning


def train(config_path, version=None, render=False, debug=False, profile=False, training=True):
    # load config
    config = load_config(config_path, version=version, render=render, debug=debug,
                         training=training)

    # run each evaluation
    error = False
    for evaluation_config in config:
        try:
            # create policy
            policy = load_policy(evaluation_config)

            # create trainer
            trainer = load_trainer(evaluation_config, policy)

            # perform evaluation
            trainer.execute(render=render, profile=profile)
        except (KeyboardInterrupt, SystemExit):
            log_warning("Program halting by request.")
            break
        except Exception as e:
            log_error(f"Evaluation failed: {e}")
            traceback.print_exc()
            error = True
            break

    # allow local runs to keep tensorboard alive
    if config.local and not error:
        input("Experiment complete.  Press enter to continue...")
