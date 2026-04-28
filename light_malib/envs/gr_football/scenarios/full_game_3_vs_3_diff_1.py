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
  diff_lvl = 1
  x_offset = 0.85 / 10 * (10 - diff_lvl)

  builder.SetBallPosition(x_offset, 0.0)

  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left


  builder.config().right_team_difficulty = diff_lvl / 10.0
  builder.config().left_team_difficulty = 1.0


  builder.SetTeam(first_team)  
  builder.AddPlayer(-1.000000, 0.420000, e_PlayerRole_GK, True, False)
  builder.AddPlayer(-0.184212 + x_offset, -0.105680, e_PlayerRole_CM)  
  builder.AddPlayer(-0.184212 + x_offset, 0.10568, e_PlayerRole_CM)  
  builder.AddPlayer(0.000000 + x_offset, 0.000000, e_PlayerRole_CF)  

  builder.SetTeam(second_team)  
  builder.AddPlayer(-1.000000, 0.420000, e_PlayerRole_GK, True, False)  
  builder.AddPlayer(-0.184212 + x_offset, -0.105680, e_PlayerRole_CM)  
  builder.AddPlayer(-0.184212 + x_offset, 0.10568, e_PlayerRole_CM)  
  builder.AddPlayer(-0.050000 + x_offset, 0.000000, e_PlayerRole_CF)  