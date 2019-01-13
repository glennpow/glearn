!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# use tf.Dataset for all experiments.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# from "trainer.py", extract "evaluator.py", "optimizer.py", etc.
# use tf.name_scope(None) around all summaries?
# concatenate all historic tensorboard data for previous versions of a model.
? refactor loss (probably needs to be in trainer, use policy.get_logits() or something)
X fix everaging of evaluation summary results
X replace GaussianLayer with CategoricalLayer
# gather multiple episodes (5+)
# empirical rewards (curiosity?)
X L2 penalty on tanh preactivation (HACK)
# transition buffer for RL should use tf, and return components as slices (producer?)
? prepare_feeds should just look at fetches, not families (requires callback from policy.run to trainer)
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# define TD/PolicyGradient inheritance.
# preparing for GPU training, some inputs/vars/operations should require:  with tf.device('/cpu:0'):
