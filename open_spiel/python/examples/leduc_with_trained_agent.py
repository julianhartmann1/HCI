# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent vs Tabular Q-Learning agents trained on Tic Tac Toe.

The two agents are trained by playing against each other. Then, the game
can be played against the DQN agent from the command line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp


FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "leduc_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")
flags.DEFINE_string("evaluation_metric", "nash_conv",
                    "Choose from 'exploitability', 'nash_conv'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/home/julian/open_spiel/trained_agents/full_trained/",
                    "Directory to save/load the agent.")



def pretty_board(time_step):
  """Returns the board in `time_step` in a human readable format."""
  info_state = time_step.observations["info_state"][0]
  x_locations = np.nonzero(info_state[9:18])[0]
  o_locations = np.nonzero(info_state[18:])[0]
  board = np.full(3 * 3, ".")
  board[x_locations] = "X"
  board[o_locations] = "0"
  board = np.reshape(board, (3, 3))
  return board


def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  logging.info("Total rewards: %s", sum_episode_rewards)
  return sum_episode_rewards / num_episodes


def analyzeHistory():
    p0_played = {"fastplay": 0, "badhand": 0, "semibluff": 0}
    p1_played = {"fastplay": 0, "badhand": 0, "semibluff": 0}
    p0_missed = {"fastplay": 0, "badhand": 0, "semibluff": 0}
    p1_missed = {"fastplay": 0, "badhand": 0, "semibluff": 0}
    with open("../../../build/python/log.txt") as log_file:
        text = log_file.readlines()
        for i in range(0, len(text), 4):
            winner = 0 if text[i].split()[2] == "P0" else 1
            p0_text = text[i + 1].split()
            if p0_text[4] == "played":
                p0_played[p0_text[2]] += 1
            else:
                p0_missed[p0_text[2]] += 1

            p1_text = text[i + 2].split()
            if p1_text[4] == "played":
                p1_played[p1_text[2]] += 1
            else:
                p1_missed[p1_text[2]] += 1

        print(p0_played)
        print(p1_played)


def evaluateBotAgainstBot(env, agent_1, agent_2, num_episodes):
    cur_agents = [agent_1, agent_2]
    reward = 0
    rewards = []
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[0]
      reward += episode_rewards
      rewards.append(reward)
    plt.plot(range(len(rewards)), rewards)
    plt.savefig("plots/agent_rewards")
    return reward / num_episodes


def main(_):
  game = "leduc_poker"
  num_players = 2
  env = rl_environment.Environment(game)
  state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]

  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }



  with tf.Session() as sess:
    agents = [
        nfsp.NFSP(sess, idx, state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]

    # for agent in agents[2:]:
    #     agent.restore("/home/benedikt/Dokumente/Uni/HCI/openspiel_saves/half_trained")

    for agent in agents:
      agent.restore(FLAGS.checkpoint_dir)
    # agents[1].restore("/home/benedikt/Dokumente/Uni/HCI/openspiel_saves/half_trained")




    # Evaluate against random agent
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions) for idx in range(num_players)
    ]

    r_mean = evaluateBotAgainstBot(env, agents[0], agents[1], 10000)
    logging.info("Mean episode rewards: %s", r_mean)

    #analyzeHistory()

    #r_mean = eval_against_random_bots(env, agents, random_agents, 10000)
    #logging.info("Mean episode rewards: %s", r_mean)

    '''if not FLAGS.iteractive_play:
      return

    # Play from the command line against the trained DQN agent.
    human_player = 1
    while True:
      logging.info("You are playing as %s", "X" if human_player else "0")
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == human_player:
          agent_out = agents[human_player].step(time_step, is_evaluation=True)
          logging.info("\n%s", agent_out.probs.reshape((3, 3)))
          logging.info("\n%s", pretty_board(time_step))
          action = command_line_action(time_step)
        else:
          agent_out = agents[1 - human_player].step(
              time_step, is_evaluation=True)
          action = agent_out.action
        time_step = env.step([action])

      logging.info("\n%s", pretty_board(time_step))

      logging.info("End of game!")
      if time_step.rewards[human_player] > 0:
        logging.info("You win")
      elif time_step.rewards[human_player] < 0:
        logging.info("You lose")
      else:
        logging.info("Draw")
      # Switch order of players
      human_player = 1 - human_player '''

if __name__ == "__main__":
  app.run(main)
  #analyzeHistory()