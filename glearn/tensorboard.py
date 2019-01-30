import time
import traceback
from glearn.utils.config import load_config
from glearn.utils.summary import SummaryWriter
from glearn.utils.log import log_error


def tensorboard(config_path, version=None):
    # load config
    config = load_config(config_path, version=version, render=False, debug=False, training=False)

    try:
        # start tensorboard
        summary = SummaryWriter(config)
        summary.start_server()

        while True:
            time.sleep(1)
    except Exception as e:
        log_error(f"Tensorboard failed: {e}")
        traceback.print_exc()
