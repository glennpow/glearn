#!/usr/bin/env python3

import click
import gym
from policies.policy_gradient import PolicyGradient
from policies.cnn import CNN
from policies.rnn import RNN
from datasets.mnist.mnist import train as mnist_dataset
from datasets.ptb.ptb import train as ptb_dataset
from datasets.sequence_tests import DigitRepeatDataset as digit_repeat_dataset
from utils.config import load_config
from utils.profile import open_profile


TEMP_DIR = "/tmp/learning"


@click.command()
@click.option("--config", "-c", "config_path", default=None)
@click.option("--env", "-e", "env_name", default=None)
@click.option("--dataset", "-d", "dataset_name", default=None)
@click.option("--policy", "-p", default="policy_gradient")
@click.option("--episodes", type=int, default=5000)
@click.option("--epochs", type=int, default=1)
@click.option("--seed", "-s", type=int, default=1)
@click.option("--batch", "batch_size", type=int, default=128)
@click.option("--evaluate", "evaluate_interval", type=int, default=5)
@click.option("--timesteps", type=int, default=35)
@click.option("--version", "-v", default=None)
@click.option("--render/--no-render", default=False)
@click.option("--profile/--no-profile", default=False)
def main(config_path, env_name, dataset_name, policy, episodes, epochs, seed, batch_size,
         evaluate_interval, timesteps, version, render, profile):
    # load config
    if config_path is not None:
        config = load_config(config_path)

    # get env or dataset
    env = None
    dataset = None
    if env_name is not None:
        # make env
        env = gym.make(env_name)
    elif dataset_name is not None:
        if dataset_name == "mnist":
            dataset = mnist_dataset(batch_size, max_count=1000)
        if dataset_name == "ptb":
            dataset = ptb_dataset(batch_size, timesteps)
        if dataset_name == "digit_repeat":
            dataset = digit_repeat_dataset(batch_size=batch_size, timesteps=timesteps)
    if env is None and dataset is None:
        print("Failed to find env or dataset to train with")
        return

    # prepare log paths
    project = env_name if env_name is not None else dataset_name
    if version is None:
        next_version = 1
        log_dir = f"{TEMP_DIR}/{project}/{next_version}"
        load_path = None
        save_path = None  # f"{log_dir}/model.ckpt"
    elif version.isdigit():
        version = int(version)
        next_version = version + 1
        log_dir = f"{TEMP_DIR}/{project}/{next_version}"
        load_path = f"{TEMP_DIR}/{project}/{version}/model.ckpt"
        save_path = f"{log_dir}/model.ckpt"
    else:
        next_version = None
        log_dir = f"{TEMP_DIR}/{project}/{version}"
        load_path = f"{TEMP_DIR}/{project}/{version}/model.ckpt"
        save_path = None
    tensorboard_path = f"{log_dir}/tensorboard/"
    profile_path = f"{log_dir}/profile" if profile else None

    # create policy
    if policy == "cnn":
        policy = CNN(env=env,
                     dataset=dataset,
                     batch_size=batch_size,
                     seed=seed,
                     # learning_rate=0.02,
                     # discount_factor=0.99,
                     load_path=load_path,
                     save_path=save_path,
                     tensorboard_path=tensorboard_path)
    elif policy == "rnn":
        policy = RNN(env=env,
                     dataset=dataset,
                     batch_size=batch_size,
                     seed=seed,
                     hidden_depth=2,  # FIXME - config...
                     # learning_rate=0.02,
                     # discount_factor=0.99,
                     load_path=load_path,
                     save_path=save_path,
                     tensorboard_path=tensorboard_path)
    else:
        policy = PolicyGradient(env=env,
                                dataset=dataset,
                                batch_size=batch_size,
                                seed=seed,
                                # learning_rate=0.02,
                                discount_factor=0.99,
                                load_path=load_path,
                                save_path=save_path,
                                tensorboard_path=tensorboard_path)

    # train policy
    policy.train(episodes=episodes, epochs=epochs, evaluate_interval=evaluate_interval,
                 render=render, profile_path=profile_path)

    if profile_path is not None:
        open_profile(profile_path)


if __name__ == "__main__":
    main()
