#!/usr/bin/env python3

import click
import time
import numpy as np
import gym
from policy_gradient import PolicyGradient
from cnn import CNN


RENDER_ENV = True
RENDER_REWARD_MIN = 5000


@click.command()
@click.option("--env", "env_name", default="SpaceInvaders-v0")
@click.option("--policy", default="policy_gradient")
@click.option("--episodes", type=int, default=5000)
@click.option("--seed", type=int, default=1)
def main(env_name, policy, episodes, seed):
    global RENDER_ENV

    # make env
    env = gym.make(env_name)
    env = env.unwrapped
    # Policy gradient has high variance, seed for reproduceability
    env.seed(seed)

    # Load checkpoint
    load_version = 0
    save_version = load_version + 1
    tmp = "/tmp/learning"
    current_tmp_dir = f"{tmp}/{env_name}/{save_version}"
    load_path = f"{tmp}/{env_name}/{load_version}/model.ckpt"
    save_path = f"{current_tmp_dir}/model.ckpt"
    tensorboard_path = f"{current_tmp_dir}/tensorboard/"

    rewards = []

    # create policy
    if policy == "cnn":
        policy = CNN(env=env,
                     learning_rate=0.02,
                     discount_factor=0.99,
                     load_path=load_path,
                     save_path=save_path,
                     tensorboard_path=tensorboard_path)
    else:
        policy = PolicyGradient(env=env,
                                learning_rate=0.02,
                                discount_factor=0.99,
                                load_path=load_path,
                                save_path=save_path,
                                tensorboard_path=tensorboard_path)

    for episode in range(episodes):
        policy.reset()
        tic = time.clock()

        while True:
            if RENDER_ENV:
                env.render()

            # rollout
            transition = policy.rollout()
            done = transition.done

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 120:
                done = True

            episode_reward = policy.episode_reward
            if episode_reward < -250:
                done = True

            if done:
                rewards.append(episode_reward)
                max_reward_so_far = np.amax(rewards)

                # train after episode
                policy.train()

                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_reward)
                print("Max reward so far: ", max_reward_so_far)

                if max_reward_so_far > RENDER_REWARD_MIN:
                    RENDER_ENV = True
                break

    plot(rewards, xlabel="Step", ylabel="Reward")


def plot(values, xlabel="X", ylabel="Y"):
    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    main()
