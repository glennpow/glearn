X show seed in evaluation start
X add metrics to ReplayBuffer
# need to remove only tensorboard summaries (not saved models)

# dropout  FIXME - implement this like batch_norm, and check all appropriate queries...
# fix tensorboard staying alive on Ctrl-C?
# from "trainer.py", extract "evaluator.py", "optimizer.py", etc.  (and could subclass dataset/env versions)
? prepare_feeds should just look at fetches, not queries (requires callback from policy.run to trainer)
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# config option to stop all sweeps after first exception
# empirical rewards for reinforcement (curiosity?)


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


### GAN

# run with sweeps of seed to test fixed evaluate noise.
# train discriminator to classify.
# train VAE or CAE in parallel.
  * 3-way classify for discriminator?
  * the Generator could use combined (weighted?) loss.


### QUESTIONS

# how does batch-size fit into RL?
  * are single episodes chunked up into steps of batch-size?
  * are they shown batch-size number of whole-episodes (still possibly chunked)?
# replay buffers
  * detail "chunks"
  * does python/numpy handle buffer swaps in memory well?  (eg. calling store_episode)
# explain difference between neglogp and softmax functions (all of them)
