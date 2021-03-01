import numpy as np

# Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
# Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
# Engine can't work with less than 50% power.
# Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.


class Shield:
    def __init__(self, thresholds_main_engine=0.9):
        self.thresholds_main_engine = thresholds_main_engine

    def shield_action(self, action):

        action_main_engine = np.clip(action[0], -self.thresholds_main_engine,
                                     self.thresholds_main_engine)

        action = [action_main_engine, action[1]]
        return action


if __name__ == '__main__':
    shield = Shield(thresholds_main_engine=0.9)
    a = np.array([0, 1])
    action = shield_action(a)
    print(action)