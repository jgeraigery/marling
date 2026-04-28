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

from light_malib.registry import registry


@registry.registered(registry.STOPPER, "win_rate_stopper")
class WinRateStopper:
    def __init__(self, **kwargs):
        self.min_win_rate = kwargs["min_win_rate"]
        # self.max_steps = kwargs["max_steps"]

    def stop(self, **kwargs):
        step = kwargs["step"]
        win_rate = kwargs["win"]
        # if step >= self.max_steps or win_rate >= self.min_win_rate:
        if win_rate >= self.min_win_rate:
            return True
        return False
    
@registry.registered(registry.STOPPER, "curriculum_win_rate_stopper")
class CurriculumWinRateStopper:
    def __init__(self, **kwargs):
        self.min_win_rate = kwargs["curriculum_min_win_rate"]

    def stop(self, **kwargs):
        win_rate = kwargs["win"]
        if win_rate >= self.min_win_rate:
            return True
        return False
    
@registry.registered(registry.STOPPER, "challenge_win_rate_stopper")
class ChallengeWinRateStopper:
    def __init__(self, **kwargs):
        self.min_win_rate = kwargs["challenge_min_win_rate"]

    def stop(self, **kwargs):
        win_rate = kwargs["win_against_recent"]
        if win_rate >= self.min_win_rate:
            return True
        return False
    
@registry.registered(registry.STOPPER, "generalize_win_rate_stopper")
class GeneralizeWinRateStopper:
    def __init__(self, **kwargs):
        self.min_win_rate = kwargs["generalize_min_win_rate"]

    def stop(self, **kwargs):
        win_rate = kwargs["win"]
        if win_rate >= self.min_win_rate:
            return True
        return False

