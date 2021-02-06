import numpy as np

# Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
# Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
# Engine can't work with less than 50% power.
# Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.


class Shield:
    def __init__(self,
                 thresholds_main_engine=0.9,
                 thresholds_left_engine=-.8,
                 thresholds_right_engine=.8):
        self.thresholds_main_engine = thresholds_main_engine
        self.thresholds_left_engine = thresholds_left_engine
        self.thresholds_right_engine = thresholds_right_engine

    def shield_action(self, action):

        action_main_engine = np.clip(action[0], -self.thresholds_main_engine,
                                     self.thresholds_main_engine)

        action_lateral_engines = np.clip(action[1],
                                         self.thresholds_left_engine,
                                         self.thresholds_right_engine)

        action = [action_main_engine, action_lateral_engines]
        return action
