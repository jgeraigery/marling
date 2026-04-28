# coding=utf-8
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

from gfootball.scenarios import *


def build_scenario(builder):
  builder.config().game_duration = 3000
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False

  # Set the ball position in proportion to the original position and according to the difficulty level
  diff_lvl = 2
  x_offset = 0.85 / 10 * (10 - diff_lvl)

  builder.SetBallPosition(x_offset, 0.0)

  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left


  builder.config().right_team_difficulty = 0.15
  builder.config().left_team_difficulty = 1.0


  builder.SetTeam(first_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, False, False)
  builder.AddPlayer(-0.3 + x_offset, -0.19576, e_PlayerRole_LB)
  builder.AddPlayer(-0.35 + x_offset, -0.06356, e_PlayerRole_CB)
  builder.AddPlayer(-0.35 + x_offset, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.3 + x_offset, 0.195760, e_PlayerRole_RB)
  builder.AddPlayer(-0.184212 + x_offset, -0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.267574 + x_offset, 0.000000, e_PlayerRole_CM)
  builder.AddPlayer(-0.184212 + x_offset, 0.10568, e_PlayerRole_CM)
  builder.AddPlayer(-0.010000 + x_offset, -0.21610, e_PlayerRole_LM)
  builder.AddPlayer(0.000000 + x_offset, 0.020000, e_PlayerRole_CF)
  builder.AddPlayer(0.000000 + x_offset,  -0.020000, e_PlayerRole_RM)
  builder.SetTeam(second_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, False, False)
  builder.AddPlayer(-0.3 + x_offset, -0.19576, e_PlayerRole_LB)
  builder.AddPlayer(-0.35 + x_offset, -0.06356, e_PlayerRole_CB)
  builder.AddPlayer(-0.35 + x_offset, 0.063559, e_PlayerRole_CB)
  builder.AddPlayer(-0.3 + x_offset, 0.195760, e_PlayerRole_RB)
  builder.AddPlayer(-0.184212 + x_offset, -0.105680, e_PlayerRole_CM)
  builder.AddPlayer(-0.267574 + x_offset, 0.000000, e_PlayerRole_CM)
  builder.AddPlayer(-0.184212 + x_offset, 0.10568, e_PlayerRole_CM)
  builder.AddPlayer(-0.010000 + x_offset, -0.21610, e_PlayerRole_LM)
  builder.AddPlayer(-0.050000 + x_offset, 0.000000, e_PlayerRole_CF)
  builder.AddPlayer(-0.010000 + x_offset, 0.216102, e_PlayerRole_RM)