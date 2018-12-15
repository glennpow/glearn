# fix save/load?  doesn't seem to work right.  (maybe just prepend old tensorboard data and start from last step #?)
# refactor loss (probably needs to be in trainer, use policy.get_logits() or something)
# for some inputs/vars/operations use --> with tf.device('/cpu:0'):
# fix everaging of evaluation summary results
# figure out TD and PolicyGradient inheritance
X replace GaussianLayer with CategoricalLayer
# gather multiple episodes (5+)
# empirical rewards (gamma ...)
X L2 penalty on tanh preactivation (HACK)
# how many minibatches per epoch
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# transition buffer for RL should use tf, and return components as slices
# common logger functions
# prepare_feeds should just look at fetches, not graphs (requires callback from policy.run to trainer)
