import os
import time
import numpy as np
import tensorflow as tf
from gym import Space
from gym.spaces import Discrete
from rapid.utils.printing import colorize
from viewer import AdvancedViewer
from dashboard import Dashboard


class Interface(object):
    def __init__(self, shape, size, discrete):
        self.shape = shape
        self.size = size
        self.discrete = discrete


class Transition(object):
    def __init__(self, observation, action, reward, new_observation, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation
        self.done = done
        self.info = info


class Batch(object):
    def __init__(self, data):
        if isinstance(data[0], Transition):
            self.observations = [e.observation for e in data]
            self.actions = [e.action for e in data]
            self.rewards = [e.reward for e in data]
            self.new_observations = [e.new_observation for e in data]
            self.done = [e.done for e in data]
            self.info = [e.info for e in data]
        else:
            self.observations = [d[0] for d in data]
            self.actions = [d[1] for d in data]


class Policy(object):
    def __init__(self, env=None, dataset=None, batch_size=128, seed=0,
                 load_path=None, save_path=None, tensorboard_path=None):
        self.env = env
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        if self.env is not None:
            self.env.seed(self.seed)

            self.viewer = AdvancedViewer()
            self.env.unwrapped.viewer = self.viewer

            self.input = self.parse_space(env.observation_space)
            self.output = self.parse_space(env.action_space)
        elif self.dataset is not None:
            self.input = self.parse_space(dataset.input_space)
            self.output = self.parse_space(dataset.output_space)

            self.viewer = None  # TODO dataset viewer?

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
        self.log(colorize(message))

    def parse_space(self, space):
        # get shape and discreteness of interface
        if isinstance(space, Discrete):
            return Interface((space.n, ), space.n, True)
        elif isinstance(space, Space):
            return Interface(space.shape, np.prod(space.shape), False)
        self.error(f"Invalid interface space: {space}")
        return None

    def init_model(self):
        pass

    def init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_persistence(self):
        if self.save_path is not None or self.load_path is not None:
            self.saver = tf.train.Saver()

        if self.load_path is not None:
            if os.path.exists(self.load_path):
                self.log(f"Loading model: {self.load_path}")
                self.saver.restore(self.sess, self.load_path)

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
        new_observation, reward, done, info = self.env.step(self.decode_action(action))

        # record transition
        transition = Transition(self.observation, action, reward, new_observation, done, info)
        self.transitions.append(transition)

        # update stats
        self.observation = new_observation
        self.episode_reward += transition.reward
        return transition

    def act(self, observation):
        if self.act_op is not None:
            # prepare parameters
            feed_dict = self.act_inputs(observation, {self.inputs: [observation]})

            # evaluate act graph
            return self.sess.run(self.act_op, feed_dict=feed_dict).ravel()
        return np.zeros(self.output.shape)

    def act_inputs(self, observation, feed_dict):
        return feed_dict

    def encode_action(self, action):
        # handle discrete actions
        if self.output.discrete:
            action_index = action
            action = np.zeros(self.output.shape)
            action[action_index] = 1
        return action

    def decode_action(self, action):
        # handle discrete actions
        if self.output.discrete:
            action = np.random.choice(range(len(action)), p=action)
        return action

    def get_batch(self, size=None):
        if self.reinforcement:
            return Batch(self.transitions[:size])
        else:
            return Batch(self.dataset.batch(self.batch_size))

    def optimize(self):
        # get batch
        batch = self.get_batch()

        # prepare parameters
        feed_dict = {
            self.inputs: batch.observations,
            self.outputs: batch.actions,
        }
        feed_dict = self.optimize_inputs(batch, feed_dict)

        # evaluate optimize graph on batch
        self.sess.run(self.optimize_op, feed_dict=feed_dict)

        # save model
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            self.log(f"Saved model: {save_path}")

    def optimize_inputs(self, batch, feed_dict):
        return feed_dict

    def evaluate(self, Y):
        # TODO - use?
        if self.act_op is not None:
            correct_act = tf.equal(tf.argmax(self.act_op, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_act, tf.float32))
            return accuracy
        return 0

    def train(self, episodes, max_episode_time=None, min_episode_reward=None,
              render=False):
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

                        self.log("==========================================")
                        self.log("Episode: ", episode)
                        self.log("Seconds: ", elapsed_sec)
                        self.log("Reward: ", episode_reward)
                        self.log("Max reward so far: ", max_reward_so_far)

                        # if max_reward_so_far > render_reward_min:
                        #     render = True
                        break
            else:
                # supervised learning
                self.optimize()

    def add_image(self, name, values, x, y, width, height):
        if self.viewer is not None:
            self.viewer.images[name] = (values, x, y, width, height)

    def remove_image(self, name):
        if self.viewer is not None:
            self.viewer.images.pop(name, None)
