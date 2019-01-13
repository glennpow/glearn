!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# use tf.Dataset for all experiments.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# figure out why debug mode causes tensorboard to show too many run names
# from "trainer.py", extract "evaluator.py", "optimizer.py", etc.
# use tf.name_scope(None) around all summaries?
# concatenate all historic tensorboard data for previous versions of a model.
? refactor loss (probably needs to be in trainer, use policy.get_logits() or something)
# gather multiple episodes for RL (5+)
# empirical rewards (curiosity?)
? prepare_feeds should just look at fetches, not families (requires callback from policy.run to trainer)
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# define TD/PolicyGradient inheritance.
X preparing for GPU training, some inputs/vars/operations should require:  with tf.device('/cpu:0'):
