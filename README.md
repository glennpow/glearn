## Experiments

### Space Invaders (Policy Gradient)
./scripts/glearn train space_invaders_pg

### Space Invaders (CNN)
./scripts/glearn train space_invaders

### MNIST (CNN)
./scripts/glearn train mnist

### MNIST (VAE)
./scripts/glearn train mnist_vae

### MNIST (Conditional VAE)
./scripts/glearn train mnist_cvae

### MNIST (GAN)
./scripts/glearn train mnist_gan

### MNIST (VAE-GAN)
./scripts/glearn train mnist_vae_gan

### MNIST (DCGAN)
./scripts/glearn train mnist_dcgan

### CIFAR-10 (CNN)
./scripts/glearn train cifar10

### Digit Repeat (RNN)
./scripts/glearn train digit_repeat

### PTB (RNN)
./scripts/glearn train ptb

### Mountain Car (DQN)
./scripts/glearn train mountain_car_dqn

### Mountain Car (A2C)
./scripts/glearn train mountain_car_a2c

### Mountain Car (SAC)
./scripts/glearn train mountain_car_sac


## Rendering

Rendering is disabled during training by default, and enabled by default during evaluation.
You can enable rendering with:
./scripts/glearn train ... --render


## Debugging

You can enable debugging information with:
./scripts/glearn train ... --debug

There are also various debug options that can be enabled in the experiment config.
The `_base.yaml` file has most of these listed, which can be imported into another config.

This environment variable can be used to log TF memory info:
`TF_CPP_MIN_VLOG_LEVEL`=3


## Versions

./scripts/glearn train ... --version=VERSION
VERSION is which params to load.  It can be an int or string.  If int, then autosaves to same VERSION.


## Profiling

./scripts/glearn train ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100
