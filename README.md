### Policy Gradient

## SpaceInvaders
python train.py --env=SpaceInvaders-v0 --policy=policy_gradient --render


### CNN

## SpaceInvaders CNN
python train.py --env=SpaceInvaders-v0 --policy=cnn --render

## MNIST
python train.py --dataset=mnist --policy=cnn --epochs=40 --batch=128 --evaluate=2


### RNN (all failing)

## Digit Repeat
python train.py --dataset=digit_repeat --policy=rnn --epochs=260 --evaluate=10 --batch=100 --timesteps=20 --render

## PTB
python train.py --dataset=ptb --policy=rnn --epochs=60 --evaluate=5 --batch=20
max_count=1000 ?


## Rendering
python train.py ... --render


### Versions
python train.py ... --version=VERSION
VERSION is which params to load.  It can be an int or string.  If int, then autosaves to VERSION+1


### Profiling
python train.py ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100

