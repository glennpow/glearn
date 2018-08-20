#!/usr/bin/env python3

import click
import gym
from policy_gradient import PolicyGradient
from cnn import CNN
from mnist import train as mnist_dataset


TEMP_DIR = "/tmp/learning"


@click.command()
@click.option("--mode", "-m", default="rl")
@click.option("--env", "-e", "env_name", default=None)
@click.option("--dataset", "-d", "dataset_name", default=None)
@click.option("--policy", "-p", default="policy_gradient")
@click.option("--episodes", type=int, default=5000)
@click.option("--seed", "-s", type=int, default=1)
@click.option("--batch", type=int, default=128)
@click.option("--evaluate", "evaluate_interval", type=int, default=5)
@click.option("--version", "-v", type=int, default=0)
@click.option("--render/--no-render", default=False)
def main(mode, env_name, dataset_name, policy, episodes, seed, batch, evaluate_interval,
         version, render):
    # get env or dataset
    env = None
    dataset = None
    if env_name is not None:
        # make env
        env = gym.make(env_name)
    elif dataset_name is not None:
        if dataset_name == "mnist":
            dataset = mnist_dataset(f"{TEMP_DIR}/data/mnist", max_count=1000)
    if env is None and dataset is None:
        print("Failed to find env or dataset to train with")
        return

    # prepare log paths
    load_version = version
    save_version = load_version + 1
    project = env_name if env_name is not None else dataset_name
    current_tmp_dir = f"{TEMP_DIR}/{project}/{save_version}"
    load_path = f"{TEMP_DIR}/{project}/{load_version}/model.ckpt"
    save_path = f"{current_tmp_dir}/model.ckpt"
    tensorboard_path = f"{current_tmp_dir}/tensorboard/"

    # create policy
    if policy == "cnn":
        policy = CNN(env=env,
                     dataset=dataset,
                     batch_size=batch,
                     seed=seed,
                     learning_rate=0.02,
                     # discount_factor=0.99,
                     load_path=load_path,
                     save_path=save_path,
                     tensorboard_path=tensorboard_path)
    else:
        policy = PolicyGradient(env=env,
                                dataset=dataset,
                                batch_size=batch,
                                seed=seed,
                                learning_rate=0.02,
                                discount_factor=0.99,
                                load_path=load_path,
                                save_path=save_path,
                                tensorboard_path=tensorboard_path)

    # train policy
    policy.train(episodes=episodes, evaluate_interval=evaluate_interval, render=render)


if __name__ == "__main__":
    main()
