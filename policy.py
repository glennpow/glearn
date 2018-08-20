import os
import time
import numpy as np
import tensorflow as tf
from printing import colorize, print_tabular
from interface import Interface
from dataset import Transition, transition_batch
from viewer import AdvancedViewer


class Policy(object):
    def __init__(self, env=None, dataset=None, batch_size=None, seed=0, deterministic=None,
                 load_path=None, save_path=None, tensorboard_path=None):
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.deterministic = deterministic
        if self.env is not None:
            self.env.seed(self.seed)

            self.viewer = AdvancedViewer()
            self.env.unwrapped.viewer = self.viewer

            self.input = Interface(env.observation_space)
            self.output = Interface(env.action_space)

            if self.deterministic is None:
                self.deterministic = False  # TODO - can we infer this from env?
        elif self.dataset is not None:
            self.input = dataset.input
            self.output = dataset.output

            self.viewer = None  # TODO dataset viewer?

            if self.deterministic is None:
                self.deterministic = self.dataset.deterministic

        self.load_path = load_path
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        self.act_op = None
        self.optimize_op = None

        self.observation = None
        self.transitions = []
        self.episode_reward = 0

        self.layers = {}

        self.dashboard = None

        self.init_viewer()
        self.init_model()
        self.init_session()
        self.init_persistence()
        self.init_tensorboard()

    def log(self, *args):
        print(*args)

    def error(self, message):
        self.log(colorize(message, "red"))

    def init_model(self):
        pass

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_persistence(self):
        if self.save_path is not None or self.load_path is not None:
            self.saver = tf.train.Saver()

        if self.load_path is not None:
            if os.path.exists(f"{self.load_path}.index"):
                try:
                    self.log(f"Loading model: {self.load_path}")
                    self.saver.restore(self.sess, self.load_path)
                except Exception as e:
                    self.error(str(e))

        if self.save_path is not None:
            self.log(f"Preparing to save model: {self.save_path}")
            save_dir = os.path.dirname(self.save_path)
            os.makedirs(save_dir, exist_ok=True)
            # TODO - clear old dir?

    def init_tensorboard(self):
        if self.tensorboard_path is not None:
            self.log(f"Tensorboard log directory: {self.tensorboard_path}")
            tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)

    def init_viewer(self):
        pass

    @property
    def reinforcement(self):
        return self.env is not None

    @property
    def supervised(self):
        return self.dataset is not None

    def get_layer_count(self, type_name=None):
        if type_name is None:
            return len(self.layers)
        else:
            if type_name in self.layers:
                return len(self.layers[type_name])
        return 0

    def get_layer(self, type_name, index=0):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
            if index < len(type_layers):
                return type_layers[index]
        return None

    def add_layer(self, type_name, layer):
        if type_name in self.layers:
            type_layers = self.layers[type_name]
        else:
            type_layers = []
            self.layers[type_name] = type_layers
        type_layers.append(layer)

    def reset(self):
        if self.env is not None:
            self.observation = self.env.reset()
            self.transitions = []
            self.episode_reward = 0

    def render(self, mode="human"):
        if self.env is not None:
            self.env.render(mode=mode)

    def rollout(self):
        # get action
        action = self.act(self.observation)

        # perform action
        new_observation, reward, done, info = self.env.step(self.output.decode(action))

        # record transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def act(self, observation):
        if self.act_op is not None:
            ops = self.act_ops([self.act_op])

            # prepare parameters
            feed_dict = self.act_inputs(observation, {self.inputs: [observation]})

            # evaluate act graph
            return self.sess.run(ops, feed_dict=feed_dict).ravel()
        return np.zeros(self.output.shape)

    def act_ops(self, ops):
        return ops

    def act_inputs(self, observation, feed_dict):
        return feed_dict

    def get_batch(self):
        if self.reinforcement:
            return transition_batch(self.transitions[:self.batch_size])
        else:
            return self.dataset.batch(self.batch_size)

    def optimize(self, evaluating=False, saving=True):
        # get batch
        batch = self.get_batch()

        ops = self.optimize_ops([self.optimize_op])

        # prepare optimization parameters
        feed_dict = {
            self.inputs: batch.inputs,
            self.outputs: batch.outputs,
        }
        feed_dict = self.optimize_inputs(batch, feed_dict)

        # evaluate optimize graph on batch
        self.sess.run(ops, feed_dict=feed_dict)

        if evaluating and self.evaluate_op is not None and self.loss_op is not None:
            # prepare evaluation parameters
            feed_dict = self.act_inputs(batch, feed_dict)

            # run evaluate graph
            loss, acc = self.sess.run([self.loss_op, self.evaluate_op], feed_dict=feed_dict)

            print_tabular({
                "loss": loss,
                "accuracy": acc,
            })

        # save model
        if saving and self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            self.log(f"Saved model: {save_path}")

    def optimize_ops(self, ops):
        return ops

    def optimize_inputs(self, batch, feed_dict):
        return feed_dict

    def train(self, episodes, max_episode_time=None, min_episode_reward=None,
              render=False, evaluate_interval=5):
        episode_rewards = []

        for episode in range(episodes):
            self.reset()
            tic = time.time()

            if self.reinforcement:
                # reinforcement learning
                while True:
                    if render:
                        self.render()

                    # rollout
                    transition = self.rollout()
                    done = transition.done

                    # episode time
                    toc = time.time()
                    elapsed_sec = toc - tic
                    if max_episode_time is not None:
                        # episode timeout
                        if elapsed_sec > max_episode_time:
                            done = True

                    # episode performance
                    episode_reward = self.episode_reward
                    if min_episode_reward is not None:
                        # episode poor performance
                        if episode_reward < min_episode_reward:
                            done = True

                    if done:
                        episode_rewards.append(self.episode_reward)
                        # self.dashboard_plot("Reward", episode_rewards)
                        max_reward_so_far = np.amax(episode_rewards)

                        # optimize after episode
                        self.optimize()

                        # self.log("==========================================")
                        # self.log("Episode: ", episode)
                        # self.log("Seconds: ", elapsed_sec)
                        # self.log("Reward: ", episode_reward)
                        # self.log("Max reward so far: ", max_reward_so_far)
                        print_tabular({
                            "episode": episode,
                            "time": elapsed_sec,
                            "reward": episode_reward,
                            "max_reward": max_reward_so_far,
                        })

                        # if max_reward_so_far > render_reward_min:
                        #     render = True
                        break
            else:
                # supervised learning
                evaluating = episode % evaluate_interval == 0
                saving = evaluating
                self.optimize(evaluating=evaluating, saving=saving)

    def get_viewer_size(self):
        if self.viewer is not None:
            return (self.viewer.width, self.viewer.height)
        return (0, 0)

    def add_image(self, name, values, **kwargs):
        if self.viewer is not None:
            self.viewer.add_image(name, values, **kwargs)

    def remove_image(self, name):
        if self.viewer is not None:
            self.viewer.remove_image(name)
