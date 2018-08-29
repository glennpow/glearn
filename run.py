"""
A wrapper command around train.py

## SpaceInvaders CNN
python train.py --env=SpaceInvaders-v0 --policy=cnn --version=X --render

## MNIST
python train.py --dataset=mnist --policy=cnn --version=X --epochs=40 --batch=128 --evaluate=2
python train.py --dataset=mnist --policy=cnn --version=X --epochs=40 --batch=128 --evaluate=2 --render

## PTB
python train.py --dataset=ptb --policy=rnn --epochs=60 --evaluate=5 --batch=20 --render

## PROFILE
python train.py ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100

"""

# TODO FIXME - add config files first
