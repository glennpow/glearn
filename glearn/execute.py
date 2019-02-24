import traceback
import bdb
from glearn.policies import load_policy
from glearn.policies.random import RandomPolicy
from glearn.trainers import load_trainer
from glearn.utils.config import load_config
from glearn.utils.log import log, log_error, log_warning
from glearn.utils.printing import getch


def execute(config_path, training, version=None, render=False, debug=False, profile=False,
            random=False):
    # load config
    config = load_config(config_path, version=version, render=render, debug=debug,
                         training=training)

    # run each evaluation
    error = False
    for evaluation_config in config:
        try:
            # create policy
            if random:
                policy = RandomPolicy(evaluation_config)
            else:
                policy = load_policy(evaluation_config)

            # create trainer
            trainer = load_trainer(evaluation_config, policy)

            # perform evaluation
            trainer.execute(render=render, profile=profile)
        except (KeyboardInterrupt, SystemExit, bdb.BdbQuit):
            log_warning("Evaluation halted by request.")
            break
        except Exception as e:
            log_error(f"Evaluation failed: {e}")
            traceback.print_exc()
            error = True
            break

    # allow local runs to keep tensorboard alive
    if config.local and not error:
        log("Experiment complete.  Press any key to terminate...")
        getch()
