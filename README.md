## Experiments

### SpaceInvaders (CNN)
./scripts/glearn train space_invaders

### SpaceInvaders (Policy Gradient)
./scripts/glearn train space_invaders_pg

### Curiosity (Policy Gradient)
./scripts/glearn train curiosity

### MNIST (CNN)
./scripts/glearn train mnist

### Digit Repeat (RNN)
./scripts/glearn train digit_repeat

### PTB (RNN)
./scripts/glearn train ptb


## Rendering

Rendering is enabled by default.  You can disable rendering with:
./scripts/glearn train ... --no-render


## Debugging

You can enable debugging information with:
./scripts/glearn train ... --debug


## Versions

./scripts/glearn train ... --version=VERSION
VERSION is which params to load.  It can be an int or string.  If int, then autosaves to VERSION+1


## Profiling

./scripts/glearn train ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100

