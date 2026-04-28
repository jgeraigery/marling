# Modified portions of the file are Copyright (c) 2026 Electronic Arts Inc.

# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
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

from cv2 import NONE_POLISHER
from matplotlib.pylab import f
from light_malib.utils.logger import Logger
import numpy as np

class Rewarder:
    def __init__(self, reward_config, num_players_per_team=11) -> None:
        self.player_last_hold_ball = -1
        self.last_ball_owned_team = -1
        self.reward_config = reward_config
        self.cumulative_shot_reward = None

        self.num_players_per_team = num_players_per_team

        self.offense_r_encoder = attack_r()
        self.defense_r_encoder = defense_r()
        self.default_r_encoder = default_r()

        ## Create an array of distances to ball for each player over time (max time step be 7000) and I want the default value of the np array to be inf
        # Because I want to get the min distance to the ball for each player over time
        # self.dist_to_ball = np.full((num_players_per_team, 7000), np.inf)

        # self.repeated_action_penalty = RepeatedActionPenalty(num_players_per_team=num_players_per_team)
    def reset_reward_state(self):
        self.offense_r_encoder.reward_for_pass_team = None

    def calc_reward(self, rew, state, all_states, team=0):
        """
        'score', 'left_team_active', 'right_team_roles', 'right_team_active',
        'right_team_yellow_card', 'left_team_direction', 'right_team_direction',
        'ball_owned_player', 'ball_owned_team', 'right_team_tired_factor', 'steps_left',
        'right_team', 'left_team_yellow_card', 'left_team_tired_factor', 'game_mode',
        'left_team_roles', 'ball', 'ball_rotation', 'left_team', 'ball_direction',
        'designated', 'active', 'sticky_actions'
        """

        obs = state.obs
        prev_obs = state.prev_obs
        action = state.action

        all_actions = [s.action for s in all_states]

        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]

        if obs["ball_owned_team"] != -1:
            self.last_ball_owned_team = obs["ball_owned_team"]

        if self.reward_config is None:  # default reward shaping
            raise NotImplementedError
            reward = (
                5 * win_reward(obs)
                + 5.0 * preprocess_score(obs, rew, self.player_last_hold_ball)
                + 0.03 * ball_position_reward(obs, self.player_last_hold_ball)
                + yellow_reward(prev_obs, obs)
                - 0.003 * min_dist_reward(obs)
            )

        else:

            active_player = obs["active"]

            if self.cumulative_shot_reward is None:
                self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])

            shot_reward = [0] * len(obs["left_team_roles"])

            if prev_obs["score"][1] < obs["score"][1]:
                self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])
            elif prev_obs["score"][0] < obs["score"][0]:
                shot_reward[active_player] = self.cumulative_shot_reward[active_player]
                self.cumulative_shot_reward[active_player] = 0
            else:
                if action == 12:
                    self.cumulative_shot_reward[active_player] += 1

            single_shot_reward = [0] * len(obs["left_team_roles"])
            if action == 12:
                single_shot_reward[obs["active"]] += 1

            # Normalize the official reward
            rew = rew / (len(obs["left_team"]) - 1)

            # NOTE: With reward coefficient 1.0 and 10 players, the official reward/penalty is 1.0, -1.0, or 0.0, based on scoring a goal
            reward = (
                    self.reward_config["official_reward"] * rew
            )

            gr = 0.0
            yr = 0.0
            wr = 0.0
            mdr = 0.0
            obp = 0.0
            ppp = 0.0
            pr = 0.0
            hbr = 0.0
            tdb = 0.0
            dbr = 0.0

            num_players = len(obs["left_team"]) - 1

            # NOTE: With reward coefficient 0.001 and 10 players, the range of reward/penalty is [-0.01, 0.01] 
            if self.reward_config.get('out_of_bound_penalty', None) is not None:
                obp = self.reward_config['out_of_bound_penalty'] * out_of_bounds_penalty(obs) 
                assert np.abs(obp) >= 0.0 and np.abs(obp) <= self.reward_config['out_of_bound_penalty'] * 1.0 * num_players, f'reward/penalty: {obp}'
                reward += obp

            # NOTE: With reward coefficient 0.001 and 10 players, the range of reward/penalty is [-0.001, 0.001]
            # NOTE: 45 is the number of pairs of players which can be close together
            if self.reward_config.get('gather_penalty', None) is not None:
                ppp = self.reward_config['gather_penalty'] * player_proximity_penalty(obs)
                num_pairs = num_players * (num_players - 1) // 2
                assert np.abs(ppp) >= 0.0 and np.abs(ppp) <= self.reward_config['gather_penalty'] * (1.0 / num_pairs) * num_pairs, f'reward/penalty: {ppp}'
                reward += ppp

            # NOTE: With reward coefficient 0.05 and 10 players, the reward/penalty is either 0.0 or 0.05
            if self.reward_config.get('pass_reward', None) is not None:
                pr = self.reward_config['pass_reward'] * self.offense_r_encoder.pass_reward(obs, action, team)
                assert pr == 0.0 or np.abs(pr) == self.reward_config['pass_reward'] * self.offense_r_encoder.good_pass_reward, f'reward/penalty: {pr}'
                reward += pr

            # NOTE: With reward coefficient 0.1 and 10 players, the range of reward/penalty is [-0.0001, 0.0001] 
            if self.reward_config.get('hold_ball', None) is not None:
                hbr = self.reward_config['hold_ball'] * hold_ball_reward(obs)
                assert np.abs(hbr) == 0.0 or np.abs(hbr) == self.reward_config['hold_ball'] * 0.001, f'reward/penalty: {hbr}'
                reward += hbr
            
            if self.reward_config.get('goal_reward', None) is not None:
                gr = self.reward_config['goal_reward'] * goal_reward(prev_obs, obs)
                assert np.abs(gr) == 0.0 or np.abs(gr) == self.reward_config['goal_reward']
                reward += gr

            # TODO: Add assertion to control reward range for yellow cards, win_reward, and min_dist_reward
            if self.reward_config.get('yellow_reward', None) is not None:
                yr = self.reward_config['yellow_reward'] * yellow_reward(prev_obs, obs)
                reward += yr

            if self.reward_config.get('win_reward', None) is not None:
                wr = self.reward_config['win_reward'] * win_reward(obs)
                reward += wr

            if self.reward_config.get('min_dist_reward', None) is not None:
                mdr = self.reward_config['min_dist_reward'] * min_dist_reward(obs)
                reward += mdr
            
            if self.reward_config.get('total_dist_to_ball_reward', None) is not None:
                tdb += self.reward_config['total_dist_to_ball_reward'] * total_dist_to_ball_reward(obs)
                assert tdb >= 0 and tdb <= obs["left_team"][1:].shape[0], f'reward: {tdb}'
                reward += tdb
            
            if self.reward_config.get('different_action_reward', None) is not None:
                dbr += self.reward_config['different_action_reward'] * different_behavior_reward(obs, all_actions)
                assert dbr >= 0 and dbr <= obs["left_team"][1:].shape[0], f'reward: {dbr}'
                reward += dbr

            # if obp != 0.0:
            #    Logger.info(f'Out of bounds penalty: {obp}')
            # if ppp != 0.0:
            #     Logger.info(f'Player proximity penalty: {ppp}')
            # if pr != 0.0:
            #    Logger.info(f'Pass reward: {pr}')
            # if hbr != 0.0:
            #     Logger.info(f'Hold ball reward: {hbr}')
            # if gr != 0.0:
            #     Logger.info(f'Goal reward: {gr}')
            # if yr != 0.0:
            #     Logger.info(f'Yellow card reward: {yr}')
            # if wr != 0.0:
            #     Logger.info(f'Win reward: {wr}')
            # if mdr != 0.0:
            #    Logger.info(f'Minimum distance reward: {mdr}')
            # if rew != 0.0:
            #     Logger.info(f'Official reward: {rew}')
            
            # Logger.info(f'Final reward: {reward}')

        return reward

def out_of_bounds_penalty(obs, penalty_factor=-1.0):
    penalty = 0.0  
  
    field_x_limits = [-1.0, 1.0]  
  
    field_y_limits = [-0.42, 0.42]

    # NOTE: obs["left_team"] gives us the positions of "our" players, and 
    # obs["right_team"] gives us the positions of the opponent players

    # For each team
    for team_name in ["left_team", "right_team"]:
        # For each player in the team except the goalkeeper
        for player_pos in obs[team_name][1:]:  
            if player_pos[0] < field_x_limits[0] or player_pos[0] > field_x_limits[1]:  
                # If a player of our team is out of bounds, we penalize ourselves
                if team_name == "left_team":
                    penalty += penalty_factor
                # If a player of the opponent team is out of bounds, we reward ourselves
                else:
                    penalty += -penalty_factor
                continue
            if player_pos[1] < field_y_limits[0] or player_pos[1] > field_y_limits[1]:  
                # If a player of our team is out of bounds, we penalize ourselves
                if team_name == "left_team":
                    penalty += penalty_factor
                # If a player of the opponent team is out of bounds, we reward ourselves
                else:
                    penalty += -penalty_factor
                continue
  
    return penalty 

def player_proximity_penalty(obs, threshold=0.05):  
    num_players = len(obs["left_team"]) - 1
    num_pairs = num_players * (num_players - 1) // 2
    penalty_factor = -(1.0 / num_pairs)
    penalty = 0.0  

    for team_name in ["left_team", "right_team"]:
        team_pos = obs[team_name]
        for i in range(1, num_players):  
            for j in range(i + 1, num_players):  
                distance = np.linalg.norm(team_pos[i] - team_pos[j])
                if distance < threshold:
                    # If our team is too close to each other, we penalize ourselves
                    if team_name == "left_team":
                        penalty += penalty_factor
                    # If the opponent team is too close to each other, we reward ourselves
                    else:
                        penalty += -(penalty_factor)
    
    return penalty  

def role_based_r(pre_obs, obs):
    team_goal_weight = {0: 0.2, 1: 0.2, 2: 0.5, 3: 0.5, 5: 0.7, 6: 1, 7: 1, 9: 1}
    team_lose_weight = {0: 1, 1: 1, 2: 0.7, 3: 0.7, 5: 0.5, 6: 0.2, 7: 0.2, 9: 0.2}

    current_role = obs["left_team_roles"][obs["active"]]

    r = 0

    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        r -= team_lose_weight[current_role]

    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        r += team_goal_weight[current_role]

    return r


def pure_goal(pre_obs, obs):
    r = 0
    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        r += 1.0
    return r


def pure_lose_goal(pre_obs, obs):
    penalty = 0.0
    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        penalty -= 1.0

    return penalty

# class ActionPenalty:  
#     def __init__(self, num_players, max_history=5, penalty_factor=-1.0):  
#         self.num_players = num_players  
#         self.max_history = max_history  
#         self.penalty_factor = penalty_factor  
#         self.action_history = [[] for _ in range(num_players)]  
  
#     def update_history(self, actions):  
#         for i, action in enumerate(actions):  
#             self.action_history[i].append(action)  
#             if len(self.action_history[i]) > self.max_history:  
#                 self.action_history[i].pop(0)  
  
#     def calculate_penalty(self):
#         print(f'Calculating penalty for action repeat/alternating')
#         penalty = 0.0  
#         for player_actions in self.action_history:  
#             if len(player_actions) >= 2:  
#                 if len(set(player_actions)) == 1:  # Repeating the same action  
#                     penalty += self.penalty_factor  
#                 elif len(set(player_actions)) == 2 and len(player_actions) >= 4:  
#                     # Check for alternating actions  
#                     # is_alternating = True  
#                     # for i in range(2, len(player_actions)):  
#                     #     if player_actions[i] != player_actions[i % 2]:  
#                     #         is_alternating = False  
#                     #         break  
#                     # if is_alternating:  
#                     penalty += self.penalty_factor  
  
#         if penalty != 0.0:  
#             print(f"Action repeat/alternating penalty: {penalty}")

#         return penalty  

class attack_r:
    def __init__(self):
        self.lost_ball_penalty = -1
        self.lost_ball_recording = False

        self.steal_ball_reward = 1
        self.steal_ball_recording = False

        # (passing_flag, team_id), team_id: 0 for left, 1 for right
        self.passing_flag = [(False, -1) for _ in range(11)]
        self.bad_pass_penalty = -1
        self.good_pass_reward = 1

        self.single_shot_reward = 0
        self.cumulative_shot_reward = None
        self.cumulative_shot_reward_factor = 1

        self.pass_reward_list = None

        self.check_offside = False

    def r(self, obs, prev_obs, action, id):

        if "team_1" in id:
            return 0

        lost_ball_r = self.lost_possession(obs, prev_obs, current_player=obs["active"])
        pass_r = self.pass_reward(obs, action)
        shot_r = self.shot_reward(
            obs, prev_obs, current_player=obs["active"], player_action=action
        )

        return lost_ball_r + pass_r + shot_r

    def lost_possession(self, obs, prev_obs, current_player):
        """
        this will include all scenario losing the ball, including being intercepted, out-of-bound,
        offside, shot gets blocked by opponent goalkeeper
        """
        r = 0
        if prev_obs["score"][0] < obs["score"][0]:
            self.lost_ball_recording = (
                False  # change of ball ownership due to ours goal
            )
            return r

        # if obs['game_mode'] == 3:                     #change mainly dur to we offside, here we penalise offside move
        #     self.lost_ball_recording = False
        #     return r

        if self.lost_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 0:  # back to our team
                self.lost_ball_recording = False
                # can add reward here
            else:  # opponent own it
                if self.last_hold_player == 0:  # our goalkeeper lose the ball
                    self.lost_ball_recording = False

                if obs["active"] == self.last_hold_player:
                    self.lost_ball_recording = False
                    r = self.lost_ball_penalty

        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
            if (
                current_player == prev_obs["ball_owned_player"]
            ):  # current player is the last holding player
                r = self.lost_ball_penalty

        elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == -1:
            self.lost_ball_recording = True
            self.last_hold_player = prev_obs["ball_owned_player"]

        return r

    def pass_reward(self, obs, player_action, team):
        r = 0.0

        if self.reward_for_pass_team is not None:
            if self.reward_for_pass_team == team:
                r += self.good_pass_reward
            else:
                r += self.bad_pass_penalty
        
        else:
            for i, p in enumerate(self.passing_flag):
                if p[0]:  
                    # If we have the ball and the owner is not the passer
                    if obs["ball_owned_team"] == 0 and obs["ball_owned_player"] != i:
                        # If we started the pass, it is a successful pass
                        # Also if we did not start the pass, it is an interception
                        # In both cases, our team gets a reward and the other team gets a penalty
                        self.passing_flag[i] = (False, -1)
                        r += self.good_pass_reward
                        self.reward_for_pass_team = team
                    elif obs["ball_owned_team"] == -1:
                        pass
                    # If we do not have the ball
                    elif obs["ball_owned_team"] == 1 and obs["ball_owned_player"] != i:
                        # If we passed the ball, we have lost the ball
                        # And if we did not pass the ball, the other team has done a successful pass
                        # In both cases, our team gets a penalty and the other team gets a reward
                        self.passing_flag[i] = (False, -1)
                        r += self.bad_pass_penalty
                        self.reward_for_pass_team = 1 if team == 0 else 0

        # If current action is passing
        if player_action == 9 or player_action == 10 or player_action == 11:
            if (
                obs["ball_owned_team"] == 0 and
                not self.passing_flag[obs["active"]][0]
                and (obs["active"] == obs["ball_owned_player"])
            ):
                self.passing_flag[obs["active"]] = (True, team)

        return r

    def goal_pass_reward(self, obs, prev_obs, action):
        """
        reward passing only after goals
        """

        if self.pass_reward_list is None:
            self.pass_reward_list = [0] * len(obs["left_team_roles"])
        pass_reward = [0] * len(obs["left_team_roles"])

        if (
            prev_obs["score"][1] < obs["score"][1]
        ):  # opponent goal, clear the pass reward
            self.pass_reward_list = [0] * len(obs["left_team_roles"])
        elif prev_obs["score"][0] < obs["score"][0]:
            pass_reward[obs["active"]] = self.pass_reward_list[obs["active"]]
            self.pass_reward_list[obs["active"]] = 0
        else:
            if action == 9 or action == 10 or action == 11:
                self.pass_reward_list[obs["active"]] += 1

        return pass_reward[obs["active"]]

    def shot_reward(self, obs, prev_obs, current_player, player_action):
        """
        reward shotting after goals
        """

        r = 0
        if self.cumulative_shot_reward is None:
            self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])

        shot_reward = [0] * len(obs["left_team_roles"])

        if prev_obs["score"][1] < obs["score"][1]:
            self.cumulative_shot_reward = [0] * len(obs["left_team_roles"])
        elif prev_obs["score"][0] < obs["score"][0]:
            shot_reward[current_player] = self.cumulative_shot_reward[current_player]
            self.cumulative_shot_reward[current_player] = 0
        else:
            if player_action == 12:
                self.cumulative_shot_reward[current_player] += 1

                r += self.single_shot_reward

        r += shot_reward[current_player]

        return r

    def offside_pass_penalty(self, obs, prev_obs, current_player, player_action):
        # when agent pass and at least one of the teammate is at offside position, start checking, if gamemode has changed, highly likely offside
        offside_r = 0

        def is_offside(obs):
            our_team_offside = [0] * obs["left_team_roles"]
            second_last_opponent_x = sorted(obs["right_team"][:, 0])[-2]
            ball_x = obs["ball"][0]
            for player_id, left_player_pos in enumerate(obs["left_team"]):
                if (
                    obs["game_mode"] == 0
                    and left_player_pos[0] > ball_x
                    and left_player_pos[0] > second_last_opponent_x
                ):
                    our_team_offside[player_id] = 1
            return sum(our_team_offside)

        if self.check_offside:
            if obs["game_mode"] == 3:
                offside_r -= 1
                self.check_offside = False
            else:
                if obs["ball_owned_team"] == 0:
                    self.check_offside = False
                elif obs["ball_owned_team"] == 1:
                    self.check_offside = False
                else:
                    pass

        if player_action == 9 or player_action == 10 or player_action == 11:
            if is_offside(obs):
                self.check_offside = True


class defense_r:
    def __init__(self):
        self.steal_ball_reward = 1
        self.steal_ball_recording = False

    def r(self, obs, prev_obs, action, id):

        if "team_1" in id:
            return 0

        steal_ball_reward = self.get_possession(obs, prev_obs)
        min_dist_reward = self.min_dist_reward(obs)

        return steal_ball_reward + min_dist_reward

    def get_possession(self, obs, prev_obs):  # get possessing
        """
        this include some scenarios getting ball possession including intercepting, opponent out-of-bound,
        we ignore when our goalkeeper steal the ball as we dont want them to have too much pressure, and we ignore offside here
        """

        r = 0

        if prev_obs["score"][1] < obs["score"][1]:
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return r

        if (
            obs["game_mode"] == 3
        ):  # change of ball ownership from free kick, this is likely due to opponent offside
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return r

        if self.steal_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 1:
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == 0
            ):  # our goalkeeper intercept the ball
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0
                and obs["ball_owned_player"] != 0
                and obs["active"] == obs["ball_owned_player"]
            ):
                self.steal_ball_recording = False
                r += (
                    self.steal_ball_reward
                )  # only reward the agent stealing the ball (can we make it team reward?)

        if (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == 0:
            if obs["active"] == obs["ball_owned_player"]:
                r += self.steal_ball_reward

        elif (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == -1:
            self.steal_ball_recording = True
        else:
            pass

        return r

    def min_dist_reward(self, obs):

        if obs["ball_owned_team"] != 0:
            ball_position = np.array(obs["ball"][:2])
            left_team_position = obs["left_team"][1:]
            left_team_dist2ball = np.linalg.norm(
                left_team_position - ball_position, axis=1
            )
            min_dist2ball = np.min(left_team_dist2ball)
        else:
            min_dist2ball = 0.0

        return min_dist2ball


class default_r:
    def __init__(self):
        self.player_last_hold_ball = -1

    def r(self, obs, prev_obs):
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]

        win_reward = self.win_reward(obs)
        goal_reward = self.goal_reward(prev_obs, obs)
        yellow_reward = self.yellow_reward(prev_obs, obs)
        ball_pos_reward = self.ball_position_reward(obs, self.player_last_hold_ball)
        hold_ball_reward = self.hold_ball_reward(obs)
        dist_to_goal = self.dist_goal_to_line(obs)

        return (
            win_reward
            + goal_reward
            + yellow_reward
            + ball_pos_reward
            + hold_ball_reward
            + dist_to_goal
        )

    def win_reward(self, obs):
        win_reward = 0.0
        if obs["steps_left"] == 0:
            [my_score, opponent_score] = obs["score"]
            if my_score > opponent_score:
                win_reward = my_score - opponent_score
        return win_reward

    def goal_reward(self, prev_obs, obs):
        penalty = 0.0
        opponent_score_pre = prev_obs["score"][1]
        opponent_score_after = obs["score"][1]
        if opponent_score_after > opponent_score_pre:
            penalty -= 1.0

        current_score_pre = prev_obs["score"][0]
        current_score_after = obs["score"][0]
        if current_score_after > current_score_pre:
            penalty += 1.0

        return penalty

    def yellow_reward(self, prev_obs, obs):
        left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
            prev_obs["left_team_yellow_card"]
        )
        right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
            prev_obs["right_team_yellow_card"]
        )
        yellow_r = right_yellow - left_yellow
        return yellow_r

    def ball_position_reward(self, obs, player_last_hold_ball):
        ball_x, ball_y, ball_z = obs["ball"]
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        ball_position_r = 0.0
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):  # in our penalty area
            ball_position_r = -2.0
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):  #
            ball_position_r = -1.0
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_position_r = 0.0
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            ball_position_r = 2.0
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_position_r = 1.0
        else:
            ball_position_r = 0.0

        if obs["ball_owned_team"] == 0:
            if not obs["active"] == player_last_hold_ball:
                ball_position_r *= 0.5

        return ball_position_r

    def hold_ball_reward(self, obs):
        r = 0.0
        if obs["ball_owned_team"] == 0:
            r += 0.001
        elif obs["ball_owned_team"] == 1:
            r -= 0.001
        else:
            pass
        return r

    def dist_goal_to_line(self, obs):
        ball_position = np.array(obs["ball"][:2])
        dist_goal_to_line = np.linalg.norm(np.array([-1, 0]) - ball_position, axis=0)
        return dist_goal_to_line


def hold_ball_reward(obs):
    r = 0.0
    if obs["ball_owned_team"] == 0:
        r += 0.001
    elif obs["ball_owned_team"] == 1:
        r -= 0.001
    else:
        pass
    return r


def dist_goal_to_line(obs):
    ball_position = np.array(obs["ball"][:2])
    dist_goal_to_line = np.linalg.norm(np.array([-1, 0]) - ball_position, axis=0)
    return dist_goal_to_line


def player_move_reward(prev_obs, obs):
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)
    return left_position_move


def ball_possession_reward(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        if obs["active"] == player_last_hold_ball:
            return -0.2
    elif prev_obs["ball_owned_team"] == 1 and obs["ball_owned_team"] == 2:
        if obs["active"] == player_last_hold_ball:
            return 0.2
    else:
        return 0


def goal_reward(pre_obs, obs):
    penalty = 0.0
    opponent_score_pre = pre_obs["score"][1]
    opponent_score_after = obs["score"][1]
    if opponent_score_after > opponent_score_pre:
        penalty -= 1.0

    current_score_pre = pre_obs["score"][0]
    current_score_after = obs["score"][0]
    if current_score_after > current_score_pre:
        penalty += 1.0

    return penalty


def preprocess_score(obs, rew_signal, player_last_hold_ball):
    if rew_signal > 0:
        factor = 1.0 if obs["active"] == player_last_hold_ball else 0.3
    else:
        return rew_signal
    return rew_signal * factor


def lost_ball_reward(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        if obs["active"] == player_last_hold_ball:
            return -0.5
    return -0.1


def win_reward(obs):
    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
    if obs["steps_left"] == 0:
        # print("STEPS LEFT == 0!")
        [my_score, opponent_score] = obs["score"]
        # if my_score > opponent_score:
        win_reward = my_score - opponent_score
    return win_reward


def min_dist_reward(obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
    else:
        min_dist2ball = 0.0
    return min_dist2ball

def total_dist_to_ball_reward(obs):
    ball_position = np.array(obs["ball"][:2])

    # The reward of our distance to the ball
    left_team_position = obs["left_team"][1:]

    # Difference vector of target position and current position
    left_team2ball = ball_position - left_team_position

    # Normalize the distance to the ball on both x and y axis
    max_x_dist = 1.0
    max_y_dist = 0.42 * 2
    left_team2ball[:, 0] /= max_x_dist
    left_team2ball[:, 1] /= max_y_dist

    # Calculate the distance to the ball
    left_team_dist2ball = np.linalg.norm(left_team2ball, axis=1)

    # For each player, calculate max(1 - dist_t, 0) and sum this for all players
    reward_for_proximity = np.sum(np.maximum(1 - left_team_dist2ball, 0))

    return reward_for_proximity

def different_behavior_reward(obs, actions):

    # Give reward 1 if the action of each player is equal to something specific
    reward = 0
    for i, action in enumerate(actions):
        # Clip the target action to 0-8
        target_action = np.clip(i, 0, 8)
        if action == target_action:
            reward += 1

    return reward

def min_dist_individual_reward(obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
        min_player_id = (
            np.argmin(left_team_dist2ball) + 1
        )  # int(np.where(left_team_dist2ball == min_dist2ball)[0])
        if obs["active"] == min_player_id:
            return min_dist2ball
        else:
            return 0.0
    else:
        min_dist2ball = 0.0
    return min_dist2ball


def yellow_reward(prev_obs, obs):
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )

    yellow_r = -left_yellow
    return yellow_r


def ball_position_reward(obs, player_last_hold_ball):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):  # in our penalty area
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):  #
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    # if obs["ball_owned_team"] == 0:
    #     if not obs["active"] == player_last_hold_ball:
    #         ball_position_r *= 0.5

    return ball_position_r


def calc_skilled_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    highpass_r = 0
    if prev_obs["ball_owned_team"] == 1 or prev_obs["ball_owned_team"] == 0:
        if (
            obs["ball_owned_team"] == 1
            and prev_obs["ball_owned_player"] != obs["ball_owned_player"]
        ):
            highpass_r = 2.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)

    reward = (
        2.0 * win_reward
        + 20.0 * rew
        + 0.06 * ball_position_r
        + yellow_r
        + highpass_r
        + left_position_move
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_attack_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    left_position_move = np.sum((prev_obs["left_team"] - obs["left_team"]) ** 2)

    reward = (
        2.0 * win_reward
        + 20.0 * rew
        + 0.06 * ball_position_r
        + yellow_r
        + left_position_move
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_active_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    left_team_position = obs["left_team"]
    right_team_position = obs["right_team"]

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_skilled_deffend_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    if prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 1:
        ballowned_r = 1.0
    elif prev_obs["ball_owned_team"] == -1 and obs["ball_owned_team"] == 0:
        ballowned_r = 0.0
    else:
        ballowned_r = -1.0

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = (
        2.0 * win_reward + 20.0 * rew + 0.06 * ball_position_r + yellow_r + ballowned_r
    )
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward


def calc_offside_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow

    win_reward = 0.0
    if obs["steps_left"] == 0:
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = 1.0

    ### 鼓励球员运动
    # left_position_move = np.sum((prev_obs['left_team']-obs['left_team'])**2)

    reward = 2.0 * win_reward + 5.0 * rew + 0.06 * ball_position_r + 20 * yellow_r
    # reward = 5.0*win_reward + 5.0*rew + 15.0*ball_position_r + yellow_r
    # reward = 20.0*win_reward + 20.0*rew + 10.0*ball_position_r + yellow_r

    return reward
