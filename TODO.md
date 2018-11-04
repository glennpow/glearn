# for some inputs/vars/operations use --> with tf.device('/cpu:0'):
# fix everaging of evaluation summary results
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


---
Dzień dobry.  Now that I'm properly evaluating the test datasets, I clearly see the previous overfitting in my CIFAR-10 model.

2018-11-01 18:32:18.803452: step 36370, loss = 0.78 (4486.3 examples/sec; 0.029 sec/batch)
2018-11-01 18:32:19.091508: step 36380, loss = 0.94 (4443.6 examples/sec; 0.029 sec/batch)