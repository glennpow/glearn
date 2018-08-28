* decrease zoom for RNN embedding image




# total number of training samples
ipdb> raw_data_len = len(raw_data)
929589

# desired sample batch size
ipdb> batch_size
128

# desired number of samples to unroll in an epoch
ipdb> timesteps
35

# number of batches within the total samples 
ipdb> raw_batch_len = raw_data_len // batch_size
7262

# number of unrolled batches in an epoch
ipdb> raw_epoch_size = (raw_batch_len - 1) // timesteps
207

