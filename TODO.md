# fix tensorboard staying alive on Ctrl-C?
# trainer should probably be the network context, no?
# from "trainer.py", extract "evaluator.py", "optimizer.py", etc.  (and could subclass supervised/reinforcement)
? refactor loss (probably needs to be in trainer, use policy.get_logits() or something)
# empirical rewards (curiosity?)
? prepare_feeds should just look at fetches, not queries (requires callback from policy.run to trainer)
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# config to stop all sweeps after first exception


## ALMOST DONE
# use tf.Dataset for all experiments. (in datasets branch)


## SOON

# ability to set sweeps such as: "policy..learning_rate": [1, 2]  which would match all learning_rates under policy
? show Q_loss in tb


### SAC

# proper replay buffer
# optimize after many episodes   (>20k env steps)
# when time to optimize RL, could/should run multiple batches (5+).



### GAUSSIAN (WOJ)

mu is only trained from state
sigma doesn't come from state, init to high value (0.1)
