### Experiments

## SpaceInvaders (CNN)
glearn/train.py configs/experiments/space_invaders.yaml --render

## SpaceInvaders (Policy Gradient)
glearn/train.py configs/experiments/space_invaders_pg.yaml --render

## Curiosity (Policy Gradient)
python glearn/train.py configs/experiments/curiosity.yaml --render

## MNIST (CNN)
python train.py --dataset=mnist --policy=cnn --epochs=40 --batch=128 --evaluate=2

## Digit Repeat (RNN)
python train.py --dataset=digit_repeat --policy=rnn --epochs=260 --evaluate=10 --batch=100 --timesteps=20 --render

## PTB (RNN)
python train.py --dataset=ptb --policy=rnn --epochs=60 --evaluate=5 --batch=20
max_count=1000 ?


### Rendering
Now the default flag is to render:
python train.py ... --render

You can disable rendering with:
python train.py ... --no-render


### Versions
python train.py ... --version=VERSION
VERSION is which params to load.  It can be an int or string.  If int, then autosaves to VERSION+1


### Profiling
python train.py ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100

