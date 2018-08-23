## SpaceInvaders CNN
python train.py --env=SpaceInvaders-v0 --policy=cnn --version=X --render

## MNIST
python train.py --dataset=mnist --policy=cnn --version=X --batch=128 --evaluate=20
python train.py --dataset=mnist --policy=cnn --version=X --batch=128 --evaluate=20 --render

## PROFILE
python train.py ... --profile
 OR
python utils/profile.py --path=/path/to/profile_100

