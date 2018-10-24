# figure out TD and PolicyGradient inheritance
X replace GaussianLayer with CategoricalLayer
# gather multiple episodes (5+)
# empirical rewards (gamma ...)
X L2 penalty on tanh preactivation (HACK)
# how many minibatches per epoch
# fix save/load?  doesn't seem to work right
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# transition buffer for RL should use tf, and return components as slices
# common logger functions
# prepare_feeds should just look at fetches, not graphs (requires callback from policy.run to trainer)
