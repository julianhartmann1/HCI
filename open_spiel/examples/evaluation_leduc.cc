// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "leduc_poker", "The name of the game to play.");
ABSL_FLAG(std::string, player1, "random", "Who controls player1.");
ABSL_FLAG(std::string, player2, "random", "Who controls player2.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_simulations, 10000, "How many simulations to run.");
ABSL_FLAG(int, num_games, 10, "How many games to play.");
ABSL_FLAG(int, max_memory_mb, 1000,
          "The maximum memory used before cutting the search short.");
ABSL_FLAG(bool, solve, true, "Whether to use MCTS-Solver.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, false, "Show the MCTS stats of possible moves.");

int games_played = absl::GetFlag(FLAGS_num_games);
int p1_games_played = games_played;
int p2_games_played = games_played;
int p1_wins = 0;
int p2_wins = 0;
int p1_semibluffs = 0;
int p1_semibluffs_missed = 0;
int p1_semibluffs_success = 0;
int p2_semibluffs = 0;
int p2_semibluffs_missed = 0;
int p2_semibluffs_success = 0;
int p1_fastplay = 0;
int p1_fastplay_missed = 0;
int p1_fastplay_success = 0;
int p2_fastplay = 0;
int p2_fastplay_missed = 0;
int p2_fastplay_success = 0;
int p1_bad_hand_fold = 0;
int p1_bad_hand_played = 0;
int p1_bad_hand_success = 0;
int p2_bad_hand_fold = 0;
int p2_bad_hand_played = 0;
int p2_bad_hand_success = 0;

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

std::unique_ptr<open_spiel::Bot> InitBot(
    std::string type, const open_spiel::Game& game, open_spiel::Player player,
    std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator) {
  if (type == "random") {
    return open_spiel::MakeUniformRandomBot(player, Seed());
  }

  if (type == "mcts") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
  }
  open_spiel::SpielFatalError("Bad player type. Known types: mcts, random");
}

open_spiel::Action GetAction(const open_spiel::State& state,
                             std::string action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  return open_spiel::kInvalidAction;
}

std::string extract_chance(std::string& chance_string) {
  std::string chance;
  if(chance_string.find("outcome:") != std::string::npos) {
    chance = chance_string.substr(chance_string.find(":") +1, 
    chance_string.size() -1);
  } else {
    throw "Chance could not be extracted!";
  }
  return chance;
}

bool check_fold(std::vector<std::string>& history) {
  for(std::string elem : history){
    if (elem.compare("Fold") == 0) {
      return true;
    }
  }
  return false;
}

int num_raise(std::vector<std::string>& history) {
  int result = 0;
  for(std::string elem : history) {
    if(elem.compare("Raise") == 0) {
      result++;
    }
  }
  return result;
}

int num_call(std::vector<std::string>& history) {
  int result = 0;
  for(std::string elem : history) {
    if(elem.compare("Call") == 0) {
      result++;
    }
  }
  return result;
}

void analyze_history(std::vector<std::string>& history, bool p1_winner) {
  /*std::cerr << std::endl;
  for(std::size_t i = 0; i < history.size(); i++){
    std::cerr << history[i] << std::endl;
  }*/

  std::string priv1 = extract_chance(history[0]);
  std::string priv2 = extract_chance(history[1]);
  std::string publ;

  std::vector<std::string> p1_bet1;
  std::vector<std::string> p2_bet1;
  std::vector<std::string> p1_bet2;
  std::vector<std::string> p2_bet2; 

  std::size_t i = 2;
  for(std::string elem : history){
    std::cerr << elem << std::endl;
  }
  std::cerr << history.size() << std::endl; 
  while (true) {
    if(!(i < history.size())){
      //std::cerr << "break on size"  << std::endl;
      break;
    } else {
      //std::cerr << "Pos " << i << " = " << history[i] << std::endl;
      if(history[i].find("outcome:") != std::string::npos) {
        //std::cerr << "break on outcome"  << std::endl;  
        break;
      } else{
        if(i % 2 == 0) {
          p1_bet1.push_back(history[i]);
        } else {
          p2_bet1.push_back(history[i]);
        }
        i++;
      }
    }
  }
  //std::cerr << "Phase 1 analyzed"  << std::endl; 
  bool fold_before_public = check_fold(p1_bet1) || check_fold(p2_bet1);
  //std::cerr << "Fold Phase 1 determined: " << fold_before_public << std::endl;
  if(!fold_before_public) {
    //std::cerr << "Beginning to analyze Phase 2" << std::endl;
    publ = extract_chance(history[i]);
    i++;
    //std::cerr << "Public card extracted" << std::endl;

    std::size_t j = 0;
    while(i + j < history.size()) {
      if(j % 2 == 0) {
        p1_bet2.push_back(history[i+j]);
      } else {
        p2_bet2.push_back(history[i+j]);
      }
      j++;
    }
  }
  /*std::cerr << "Private 1: " << priv1 << std::endl;
  std::cerr << "Private 2: " << priv2 << std::endl;
  std::cerr << "Player 1 Actions Round 1: ";
  for(std::string elem : p1_bet1){
    std::cerr << elem << ", ";
  }
  std::cerr << std::endl;

  std::cerr << "Player 2 Actions Round 1: ";
  for(std::string elem : p2_bet1){
    std::cerr << elem << ", ";
  }
  std::cerr << std::endl;
  if(!fold_before_public){
    std::cerr << "Public Card: " << publ << std::endl;
    std::cerr << "Player 1 Actions Round 2: ";
    for(std::string elem : p1_bet2){
      std::cerr << elem << ", ";
    }
    std::cerr << std::endl;
    std::cerr << "Player 2 Actions Round 2: ";
    for(std::string elem : p2_bet2){
      std::cerr << elem << ", ";
    }
    std::cerr << std::endl;

  }*/

  //std::cerr << "Phase 2 analyzed";
  if(check_fold(p1_bet1) && num_raise(p1_bet1) == 0) {
    p1_games_played--;
      //std::cerr << "Player 0 did not play" << std::endl;
  }
  if(check_fold(p2_bet1) && num_raise(p2_bet1) == 0) {
    p2_games_played--;
      //std::cerr << "Player 1 did not play" << std::endl;
  }

  if(priv1.compare("0") == 0 || priv1.compare("1") == 0) {
    if(check_fold(p1_bet1) && num_raise(p1_bet1) == 0) {
      p1_bad_hand_fold++;
    } else {
      p1_bad_hand_played++;
      if(p1_winner) {
        p1_bad_hand_success++;
      }
    }
  }

  if(priv2.compare("0") == 0 || priv2.compare("1") == 0) {
    if(check_fold(p2_bet1) && num_raise(p2_bet1) == 0) {
      p2_bad_hand_fold++;
    } else {
      p2_bad_hand_played++;
      if(!p1_winner) {
        p2_bad_hand_success++;
      }
    }
  }

  if(priv1.compare("2") == 0 || priv1.compare("3") == 0) {
    if(num_raise(p1_bet1) > 0 || (num_raise(p2_bet1) > 0 && !check_fold(p1_bet1))) {
      p1_semibluffs++;
      std::cerr << std::endl << "Player 0 played a semi-bluff." << std::endl;
      if(p1_winner){
        p1_semibluffs_success++;
      }
    } else {
      p1_semibluffs_missed++;
    }
  }

  if(priv2.compare("2") == 0 || priv2.compare("3") == 0) {
    if((num_raise(p1_bet1) > 0 && !check_fold(p2_bet1)) || num_raise(p2_bet1) > 0 ) {
      p2_semibluffs++;
      std::cerr << std::endl << "Player 1 played a semi-bluff." << std::endl;
      if(!p1_winner){
        p2_semibluffs_success++;
      }
    } else {
      p2_semibluffs_missed++;
    }
  }

  if(priv1.compare("4") == 0|| priv1.compare("5") == 0) { 
    if(check_fold(p2_bet1)){
      if(num_raise(p1_bet1) > 0) {
      p1_fastplay++;
      p1_fastplay_success++;
      std::cerr << std::endl << "Player 0 fastplayed." << std::endl;
      } else {
        p1_fastplay_missed++;
      }
    } else {
      if(num_raise(p1_bet1) > 0 && num_raise(p1_bet2) > 0) {
        p1_fastplay++;
        std::cerr << std::endl << "Player 0 fastplayed." << std::endl;
        if(p1_winner) {
          p1_fastplay_success++;
        }
      } else {
        p1_fastplay_missed++;
      }
    }
  }

  if(priv2.compare("4") == 0|| priv2.compare("5") == 0) { 
    if(check_fold(p1_bet1)){
      if(num_raise(p2_bet1) > 0) {
      p2_fastplay++;
      p2_fastplay_success++;
      std::cerr << std::endl << "Player 1 fastplayed." << std::endl;
      } else {
        p2_fastplay_missed++;
      }
    } else {
      if(num_raise(p2_bet1) > 0 && num_raise(p2_bet2) > 0) {
        p2_fastplay++;
        std::cerr << std::endl << "Player 1 fastplayed." << std::endl;
        if(!p1_winner) {
          p2_fastplay_success++;
        }
      } else {
        p2_fastplay_missed++;
      }
    }
  }

}

void print_stats() {
  float p1_played = roundf(p1_games_played/games_played*100) /100;
  float p2_played = roundf(p2_games_played/games_played*100) /100;
  std::cerr << "******************************" << std::endl << 
  "REPORT" << std::endl << "******************************" << 
  std::endl;
  std::cerr << "Player 0 played " << p1_games_played << " out of " << games_played << " games and won " << p1_wins << "." << std::endl;
  std::cerr << "The number of strong hands played aggressivly was " << p1_fastplay << "." << std::endl;
  std::cerr << "Out of that " << p1_fastplay_success << " games were won." << std::endl;
  std::cerr << p1_fastplay_missed << " chances of fastplaying strong hands were missed." << std::endl;
  std::cerr << "The number of semi-bluffs played was " << p1_semibluffs << "." << std::endl;
  std::cerr << "Out of that " << p1_semibluffs_success << " games were won." << std::endl;
  std::cerr << p1_semibluffs_missed << " chances of semi-bluffing were missed." << std::endl;
  std::cerr << "A bad hand was folded " << p1_bad_hand_fold << " times." << std::endl;
  std::cerr << "But, " << p1_bad_hand_played << " bad hands were played." << std::endl;
  std::cerr << "Out of that " << p1_bad_hand_success << " games were won." << std::endl;
  std::cerr << "Player 1 played " << p2_games_played << " out of " << games_played << " games and won " << p2_wins << "." << std::endl;
  std::cerr << "The number of strong hands played aggressivly was " << p2_fastplay << "." << std::endl;
  std::cerr << "Out of that " << p2_fastplay_success << " games were won." << std::endl;
  std::cerr << p2_fastplay_missed << " chances of fastplaying strong hands were missed." << std::endl;
  std::cerr << "The number of semi-bluffs played was " << p2_semibluffs << "." << std::endl;
  std::cerr << "Out of that " << p2_semibluffs_success << " games were won." << std::endl;
  std::cerr << p2_semibluffs_missed << " chances of semi-bluffing were missed." << std::endl;
  std::cerr << "A bad hand was folded " << p2_bad_hand_fold << " times." << std::endl;
  std::cerr << "But, " << p2_bad_hand_played << " bad hands were played." << std::endl;
  std::cerr << "Out of that " << p2_bad_hand_success << " games were won." << std::endl;
}


std::pair<std::vector<double>, std::vector<std::string>> PlayGame(
    const open_spiel::Game& game,
    std::vector<std::unique_ptr<open_spiel::Bot>>& bots, std::mt19937& rng,
    const std::vector<std::string>& initial_actions) {
  bool quiet = absl::GetFlag(FLAGS_quiet);
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;

  for (const auto& action_str : initial_actions) {
    open_spiel::Action action = GetAction(*state, action_str);
    if (action == open_spiel::kInvalidAction)
      open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

    history.push_back(action_str);
    state->ApplyAction(action);
    if (!quiet) {
      std::cerr << "Forced action" << action_str << std::endl;
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }
  }

  while (!state->IsTerminal()) {
    open_spiel::Player player = state->CurrentPlayer();
    if (!quiet) std::cerr << "player turn: " << player << std::endl;

    open_spiel::Action action;
    if (state->IsChanceNode()) {
      // Chance node; sample one according to underlying distribution.
      open_spiel::ActionsAndProbs outcomes = state->ChanceOutcomes();
      action = open_spiel::SampleAction(outcomes, rng).first;
      if (!quiet)
        std::cerr << "Sampled action: " << state->ActionToString(player, action)
                  << std::endl;
    } else if (state->IsSimultaneousNode()) {
      open_spiel::SpielFatalError(
          "MCTS not supported for games with simultaneous actions.");
    } else {
      // Decision node, ask the right bot to make its action
      action = bots[player]->Step(*state);
      if (!quiet)
        std::cerr << "Chose action: " << state->ActionToString(player, action)
                  << std::endl;
    }
    for (open_spiel::Player p = 0; p < bots.size(); ++p) {
      if (p != player) {
        bots[p]->InformAction(*state, player, action);
      }
    }
    history.push_back(state->ActionToString(player, action));
    state->ApplyAction(action);

    if (!quiet)
      std::cerr << "State: " << std::endl << state->ToString() << std::endl;
  }
  std::vector<double> returns = state->Returns();
  bool p1_winner = returns[0] > 0;
  if(p1_winner){
    p1_wins++;
  } else {
    p2_wins++;
  } 
  analyze_history(history,p1_winner);

  std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ",")
            << " Game actions: " << absl::StrJoin(history, " ") << std::endl;
  return {state->Returns(), history};
}

// Example code for using MCTS agent to play a game
int main(int argc, char** argv) {
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());  // Random number generator.

  // Create the game
  std::string game_name = absl::GetFlag(FLAGS_game);
  std::cerr << "game: " << game_name << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);

  // MCTS supports arbitrary number of players, but this example assumes
  // 2-player games.
  SPIEL_CHECK_TRUE(game->NumPlayers() <= 2);

  auto evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          absl::GetFlag(FLAGS_rollout_count), Seed());

  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  bots.push_back(InitBot(absl::GetFlag(FLAGS_player1), *game, 0, evaluator));
  bots.push_back(InitBot(absl::GetFlag(FLAGS_player2), *game, 1, evaluator));

  std::vector<std::string> initial_actions;
  for (int i = 1; i < positional_args.size(); ++i) {
    initial_actions.push_back(positional_args[i]);
  }

  std::map<std::string, int> histories;
  std::vector<double> overall_returns(2, 0);
  std::vector<int> overall_wins(2, 0);
  int num_games = absl::GetFlag(FLAGS_num_games);
  for (int game_num = 0; game_num < num_games; ++game_num) {
   /* std::vector<double> returns;
    std::vector<std::string> history;
    std::pair<std::vector<double>, std::vector<std::string>> result = [returns, history]; */
    auto [returns, history] = PlayGame(*game, bots, rng, initial_actions);
    histories[absl::StrJoin(history, " ")] += 1;
    for (int i = 0; i < returns.size(); ++i) {
      double v = returns[i];
      overall_returns[i] += v;
      if (v > 0) {
        overall_wins[i] += 1;
      }
    }
  }

  std::cerr << "Number of games played: " << num_games << std::endl;
  std::cerr << "Number of distinct games played: " << histories.size()
            << std::endl;
  std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ",")
            << std::endl;
  std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ",")
            << std::endl;

  print_stats();

  return 0;
}
