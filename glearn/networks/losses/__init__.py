from glearn.utils.reflection import get_function

from .loss import simple_loss  # noqa
from .sequence import sequence_loss  # noqa


def load_loss(definition, network, outputs, **kwargs):
    loss_function = get_function(definition) if definition is not None else simple_loss

    return loss_function(network, outputs, **kwargs)
