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
