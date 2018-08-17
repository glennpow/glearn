import os
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete


class Transition(object):
    def __init__(self, observation, action, reward, new_observation, done, info):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation
        self.done = done
        self.info = info


class Batch(object):
    def __init__(self, transitions):
        self.observations = [e.observation for e in transitions]
        self.actions = [e.action for e in transitions]
        self.rewards = [e.reward for e in transitions]
        self.new_observations = [e.new_observation for e in transitions]
        self.done = [e.done for e in transitions]
        self.info = [e.info for e in transitions]


class Policy(object):
    def __init__(self, env, load_path=None, save_path=None, tensorboard_path=None):
        self.env = env
        self.input_space = env.observation_space
        self.output_space = env.action_space

        self.input_shape, self.input_discrete = self.init_space(self.input_space)
        self.output_shape, self.output_discrete = self.init_space(self.output_space)

        self.load_path = load_path
        self.save_path = save_path
        self.tensorboard_path = tensorboard_path

        self.act_op = None
        self.train_op = None

        self.observation = None
        self.transitions = []
        self.episode_reward = 0

        self.init_model()
        self.init_session()
        self.init_persistence()
        self.init_tensorboard()

    def log(self, message):
        print(message)

    def init_space(self, space):
        # get shape and discreteness
        if isinstance(space, Discrete):
            return (space.n, ), True
        else:
            return space.shape, False

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

    def reset(self):
        self.observation = self.env.reset()
        self.transitions = []
        self.episode_reward = 0

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
            feed_dict = self.act_inputs({self.inputs: [observation]})

            # evaluate act graph
            return self.sess.run(self.act_op, feed_dict=feed_dict).ravel()
        return np.zeros(self.output_shape)

    def act_inputs(self, feed_dict):
        return feed_dict

    def encode_action(self, action):
        # handle discrete actions
        if self.output_discrete:
            action_index = action
            action = np.zeros(self.output_shape)
            action[action_index] = 1
        return action

    def decode_action(self, action):
        # handle discrete actions
        if self.output_discrete:
            action = np.random.choice(range(len(action)), p=action)
        return action

    def get_batch(self, size=None):
        return Batch(self.transitions[:size])

    def train(self):
        # get batch
        batch = self.get_batch()

        # prepare parameters
        feed_dict = {
            self.inputs: batch.observations,
            self.outputs: batch.actions,
        }
        feed_dict = self.train_inputs(batch, feed_dict)

        # evaluate train graph on batch
        self.sess.run(self.train_op, feed_dict=feed_dict)

        # save model
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            self.log(f"Saved model: {save_path}")

    def train_inputs(self, batch, feed_dict):
        return feed_dict
