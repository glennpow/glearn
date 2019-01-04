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
? prepare_feeds should just look at fetches, not graphs (requires callback from policy.run to trainer)
# rename Batch to Data, and subclass Dataset from it.  Also load datasets using definitions paradigm?
# define TD/PolicyGradient inheritance.
# preparing for GPU training, some inputs/vars/operations should require:  with tf.device('/cpu:0'):









### NOTES ###


┌═════════════════════════════════════════════════════════════════════════════════════════════════════┐
│  Dataset                                                                                            │
├-----------------------------------------------------------------------------------------------------┤
│  Description  │  Dataset(CIFAR-10, total=[test:9984, train:9984], batches=[test:78, train:78]x128)  │
│  Input        │  Interface((32, 32, 3) [3072], continuous, float32, deterministic)                  │
│  Output       │  Interface((10,) [10], discrete, int64, deterministic)                              │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Trainer                                                                                            │
├-----------------------------------------------------------------------------------------------------┤
│  Description  │  PolicyGradientTrainer(supervised)                                                  │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Policy                                                                                             │
├-----------------------------------------------------------------------------------------------------┤
│  Description           │  NetworkPolicy(single-threaded)                                            │
│  Global Parameters     │  5,269,280                                                                 │
│  Trainable Parameters  │  1,756,426                                                                 │
└═════════════════════════════════════════════════════════════════════════════════════════════════════┘


┌═════════════════════════════════════════════════════════════════════════════════════════════════════┐
│  Dataset                                                                                            │
├-----------------------------------------------------------------------------------------------------┤
│  Description  │  Dataset(CIFAR-10, total=[test:9984, train:9984], batches=[test:78, train:78]x128)  │
│  Input        │  Interface((32, 32, 3) [3072], continuous, float32, deterministic)                  │
│  Output       │  Interface((10,) [10], discrete, int64, deterministic)                              │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Trainer                                                                                            │
├-----------------------------------------------------------------------------------------------------┤
│  Description  │  PolicyGradientTrainer(supervised)                                                  │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Policy                                                                                             │
├-----------------------------------------------------------------------------------------------------┤
│  Description           │  NetworkPolicy(single-threaded)                                            │
│  Global Parameters     │  5,269,280                                                                 │
│  Trainable Parameters  │  1,756,426                                                                 │
└═════════════════════════════════════════════════════════════════════════════════════════════════════┘

Started tensorboard server: 35.230.110.114
Preparing to save model: /tmp/glearn/CIFAR-10/1/model.ckpt
TensorBoard 1.12.0 at http://glearn:6006 (Press CTRL+C to quit)
┌══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════┐
│  Epoch: 1                                                                                                            │
├----------------------------------------------------------------------------------------------------------------------┤
│  global step    │  100     │  int                                                                                    │
│  training time  │  48.537  │  float                                                                                  │
│  steps/second   │  2.1984  │  float                                                                                  │
│  epoch step     │  21      │  int                                                                                    │
│  epoch time     │  18.186  │  float                                                                                  │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Inputs                                                                                                              │
├----------------------------------------------------------------------------------------------------------------------┤
│  dropout  │  1  │  int                                                                                               │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│  Evaluation                                                                                                          │
├----------------------------------------------------------------------------------------------------------------------┤
│  dense_2_0_W_loss  │  0.41042                                                                   │  float32           │
│  dense_2_1_W_loss  │  0.33122                                                                   │  float32           │
│  predict           │  [[0.0265525  0.01194015 0.16048616 ... 0.10567061 0.01265951 0.00537012]  │  ndarray(128, 10)  │
│  loss              │  2.5426                                                                    │  float32           │
│  accuracy          │  0.34375                                                                   │  float32           │
└══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════┘

total 1004680
-rw-r--r--  1 glennpowell  wheel        123 Jan  2 15:04 checkpoint
-rw-r--r--  1 glennpowell  wheel   21077128 Jan  2 15:04 model.ckpt.data-00000-of-00001
-rw-r--r--  1 glennpowell  wheel       1193 Jan  2 15:04 model.ckpt.index
-rw-r--r--  1 glennpowell  wheel  493306083 Jan  2 15:04 model.ckpt.meta
drwxr-xr-x  4 glennpowell  wheel        128 Jan  2 14:40 tensorboard

total 41488
-rw-r--r--  1 glennpowell  wheel       123 Jan  2 17:54 checkpoint
-rw-r--r--  1 glennpowell  wheel  21077128 Jan  2 17:54 model.ckpt.data-00000-of-00001
-rw-r--r--  1 glennpowell  wheel      1193 Jan  2 17:54 model.ckpt.index
-rw-r--r--  1 glennpowell  wheel    153633 Jan  2 17:54 model.ckpt.meta
drwxr-xr-x  4 glennpowell  wheel       128 Jan  2 17:54 tensorboard

total 20748
-rw-r--r-- 1 root root      123 Jan  3 05:15 checkpoint
-rw-r--r-- 1 root root 21077128 Jan  3 05:15 model.ckpt.data-00000-of-00001
-rw-r--r-- 1 root root     1193 Jan  3 05:15 model.ckpt.index
-rw-r--r-- 1 root root   154369 Jan  3 05:15 model.ckpt.meta
drwxr-xr-x 4 root root     4096 Jan  3 05:15 tensorboard

