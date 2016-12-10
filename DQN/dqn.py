import argparse
import gym
import cv2
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from lib.atari.helpers import (
    AtariEnvWrapper, atari_make_next_state, atari_make_initial_state
)
from lib.atari.q_network import QNetwork
from lib.atari.memory import Memory
from collections import deque, namedtuple


env = AtariEnvWrapper(gym.envs.make("BreakoutDeterministic-v0"))

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor():
    def process(self, state):
        return cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)


def make_epsilon_greedy_policy(q_network, num_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    # def policy_fn(observation, epsilon, step):
    #     A = np.ones(nA, dtype=float) * epsilon / nA
    #     q_values = estimator.inference(np.expand_dims(observation, 0), step)[0]
    #     best_action = np.argmax(q_values)
    #     A[best_action] += (1.0 - epsilon)
    #     return A
    # return policy_fn

    def policy_fn(state, epsilon, step):
        if random.random() >= epsilon:
            #q_values = q_network.inference(np.expand_dims(state, 0), step)
            q_values = q_network.inference(state, step)
            return np.argmax(q_values)
        else:
            return random.randrange(num_actions)
    return policy_fn


def deep_q_learning(args,
                    env,
                    q_network,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    memory,
                    train_every=4,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    record_video_every=50):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment
        q_network: Both target and behavior networks
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        memory: Replay memory object
        train_every: How often to train Q network
        discount_factor: Lambda time discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    #replay_memory = []
    memory = Memory(args, len(VALID_ACTIONS))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    total_t = 0

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_network,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    frame = env.reset()
    frame = state_processor.process(frame)
    #state = atari_make_initial_state(state)
    memory.add(frame, 0, 0, int(False))
    #for i in range(args.initial_memory_length):
    for t in itertools.count():
        #action_probs = policy(state, epsilons[total_t], total_t)
        #action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        state = memory.get_current_state()
        action = policy(state, epsilons[total_t], total_t)
        one_hot_action = np.zeros(len(VALID_ACTIONS))
        one_hot_action[action] = 1
        next_frame, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_frame = state_processor.process(next_frame)
        #next_state = atari_make_next_state(state, next_state)
        #replay_memory.append(Transition(state, one_hot_action, reward, next_state, done))
        memory.add(next_frame, action, reward, int(done))

        total_t += 1

        # Ensure that we finish current episode once we've exceeded the initial
        # memory length
        if t >= args.initial_memory_length and done:
            break
        elif done:
            frame = env.reset()
            frame = state_processor.process(frame)
            #state = atari_make_initial_state(state)
            memory.add(frame, 0, 0, int(False))

    # Record videos
    env.monitor.start(monitor_path,
                      resume=True,
                      video_callable=lambda count: count % record_video_every == 0)

    # Reset the environment
    frame = env.reset()
    frame = state_processor.process(frame)
    #state = atari_make_initial_state(state)
    memory.add(frame, 0, 0, int(False))
    loss = None

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        q_network.save_model(i_episode)
        q_network.record_params(total_t)

        # One step in the environment
        for t in itertools.count():
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_network.summary_writer.add_summary(episode_summary, total_t)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            # action_probs = policy(state, epsilon, total_t)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            state = memory.get_current_state()
            action = policy(state, epsilon, total_t)
            one_hot_action = np.zeros(len(VALID_ACTIONS))
            one_hot_action[action] = 1
            next_frame, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_frame = state_processor.process(next_frame)
            #next_state = atari_make_next_state(state, next_state)

            # # If our replay memory is full, pop the first element
            # if len(replay_memory) == replay_memory_size:
            #     replay_memory.pop(0)

            # Save transition to replay memory
            # replay_memory.append(
            #     Transition(state, one_hot_action, reward, next_state, int(done))
            # )
            memory.add(next_frame, action, reward, int(done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if total_t % train_every == 0:
                # Sample a minibatch from the replay memory
                #samples = random.sample(replay_memory, args.batch_size)
                #states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
                states, actions, rewards, next_states, dones = memory.get_batch()

                # Perform gradient descent update
                #states_batch = np.array(states_batch)
                loss = q_network.train(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones
                )

                if total_t % (10 * train_every) == 0:
                    q_network.record_state(states[0], total_t)

            total_t += 1

            if done:
                try:
                    # Reset the environment
                    frame = env.reset()
                    frame = state_processor.process(frame)
                    #state = atari_make_initial_state(state)
                    memory.add(frame, 0, 0, int(False))
                    loss = None

                    break
                except:
                    pass

            #state = next_state

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        q_network.summary_writer.add_summary(episode_summary, total_t)
        q_network.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    env.monitor.close()
    return stats


def main():
    parser = argparse.ArgumentParser('a program to train or run a deep q-learning agent')
    parser.add_argument("--screen_dims", type=tuple, help="dimensions to resize frames", default=(84,84))
    parser.add_argument("--history_length", type=int, help="number of frames in a state", default=4)

    # must set network_architecture to custom in order use custom architecture
    parser.add_argument("--conv_kernel_shapes", type=tuple,
        help="shapes of convnet kernels: ((height, width, in_channels, out_channels), (next layer))")
    # must have same length as conv_kernel_shapes
    parser.add_argument("--conv_strides", type=tuple, help="connvet strides: ((1, height, width, 1), (next layer))")
    # currently,  you must have at least one dense layer
    parser.add_argument("--dense_layer_shapes", type=tuple, help="shapes of dense layers: ((in_size, out_size), (next layer))")
    parser.add_argument("--discount_factor", type=float, help="constant to discount future rewards", default=0.99)
    parser.add_argument("--learning_rate", type=float, help="constant to scale parameter updates", default=0.00025)
    parser.add_argument("--optimizer", type=str, help="optimization method for network",
        choices=('rmsprop', 'graves_rmsprop'), default='graves_rmsprop')
    parser.add_argument("--rmsprop_decay", type=float, help="decay constant for moving average in rmsprop", default=0.95)
    parser.add_argument("--rmsprop_epsilon", type=int, help="constant to stabilize rmsprop", default=0.01)
    # set error_clipping to less than 0 to disable
    parser.add_argument("--error_clipping", type=float, help="constant at which td-error becomes linear instead of quadratic", default=1.0)
    # set gradient clipping to 0 or less to disable.  Currently only works with graves_rmsprop.
    parser.add_argument("--gradient_clip", type=float, help="clip gradients to have the provided L2-norm", default=0)
    parser.add_argument("--target_update_frequency", type=int, help="number of policy network updates between target network updates", default=10000)
    parser.add_argument("--initial_memory_length", type=int,
        help="number of randomly-generated steps to initially fill replay memory", default=50000)
    parser.add_argument("--memory_capacity", type=int, help="max number of steps to store in replay memory", default=1000000)
    parser.add_argument("--batch_size", type=int, help="number of transitions sampled from memory during learning", default=32)

    parser.add_argument("--double_dqn", help="use double q-learning algorithm in error target calculation", default=False)

    args = parser.parse_args()

    args.conv_kernel_shapes = [
        [8,8,4,32],
        [4,4,32,64],
        [3,3,64,64]]
    args.conv_strides = [
        [1,4,4,1],
        [1,2,2,1],
        [1,1,1,1]]
    args.dense_layer_shapes = [[3136, 512]]

    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    state_processor = StateProcessor()

    q_network = QNetwork(args, len(VALID_ACTIONS), experiment_dir)
    memory = Memory(args, len(VALID_ACTIONS))

    for t, stats in deep_q_learning(args=args,
                                    env=env,
                                    q_network=q_network,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=100000,
                                    memory=memory,
                                    train_every=4,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=1000000,
                                    discount_factor=0.99):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

if __name__ == "__main__":
    main()
