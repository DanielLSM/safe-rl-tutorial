import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from exploration import OUActionNoise
from rpm import Buffer, update_target

from ddpg import DDPG
from shield import Shield
from lundar_landing import LunarLanderContinuous

from safe_lunar_env import UserFeedbackShield, SafeLunarEnv


def learning_loop_per_iterations(total_episodes=1000,
                                 update_shield_per_ep=10,
                                 shield_iter_per_update=10):

    user_feedback = UserFeedbackShield()
    # modes = ['left', 'left_right', 'all']
    # assert user_feed_back_mode in modes

    problem = "LunarLanderContinuous-v2"
    env = LunarLanderContinuous()
    env = SafeLunarEnv(env)

    # if shield_provided != None:
    #     env = SafeLunarEnv(env, shield_provided)
    # else:
    #     env = SafeLunarEnv(env)

    num_states = env.observation_space.shape[0] + 1
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    ddpg_agent = DDPG(problem_name=problem,
                      num_states=num_states,
                      num_actions=num_actions,
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      total_episodes=total_episodes)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(ddpg_agent.total_episodes):

        env = SafeLunarEnv(env)

        prev_state = env.reset()
        episodic_reward = 0
        render_episodes = 1000
        render = not (ep % render_episodes)

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = ddpg_agent.policy(tf_prev_state, ddpg_agent.ou_noise)
            # Recieve state and reward from environment.
            # print(action)
            state, reward, done, info = env.step(action[0])
            if render: env.render()

            ddpg_agent.buffer.record((prev_state, action[0], reward, state))
            episodic_reward += reward

            ddpg_agent.buffer.learn(ddpg_agent.target_actor,
                                    ddpg_agent.target_critic,
                                    ddpg_agent.actor_model,
                                    ddpg_agent.critic_model,
                                    ddpg_agent.actor_optimizer,
                                    ddpg_agent.critic_optimizer)

            update_target(ddpg_agent.target_actor.variables,
                          ddpg_agent.actor_model.variables, ddpg_agent.tau)
            update_target(ddpg_agent.target_critic.variables,
                          ddpg_agent.critic_model.variables, ddpg_agent.tau)

            # End this episode when `done` is True
            if done:
                break

            for _ in range(shield_iter_per_update):
                user_feedback.update_oracle_with_last_action(
                    action[0])  ##########HEREEEEEEEEEEEE is the shield
            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    return avg_reward_list, user_feedback


def learning_loop_with_shield(total_episodes=1000,
                              shield_provided=Shield(0.9, -0.8, 0.8)):

    problem = "LunarLanderContinuous-v2"
    env = LunarLanderContinuous()
    env = SafeLunarEnv(env, shield_provided)

    # if shield_provided != None:
    #     env = SafeLunarEnv(env, shield_provided)
    # else:
    #     env = SafeLunarEnv(env)

    num_states = env.observation_space.shape[0] + 1
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    ddpg_agent = DDPG(problem_name=problem,
                      num_states=num_states,
                      num_actions=num_actions,
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      total_episodes=total_episodes)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(ddpg_agent.total_episodes):

        prev_state = env.reset()
        episodic_reward = 0
        render_episodes = 1000
        render = not (ep % render_episodes)

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = ddpg_agent.policy(tf_prev_state, ddpg_agent.ou_noise)
            # Recieve state and reward from environment.
            # print(action)
            state, reward, done, info = env.step(action[0])
            if render: env.render()

            ddpg_agent.buffer.record((prev_state, action[0], reward, state))
            episodic_reward += reward

            ddpg_agent.buffer.learn(ddpg_agent.target_actor,
                                    ddpg_agent.target_critic,
                                    ddpg_agent.actor_model,
                                    ddpg_agent.critic_model,
                                    ddpg_agent.actor_optimizer,
                                    ddpg_agent.critic_optimizer)

            update_target(ddpg_agent.target_actor.variables,
                          ddpg_agent.actor_model.variables, ddpg_agent.tau)
            update_target(ddpg_agent.target_critic.variables,
                          ddpg_agent.critic_model.variables, ddpg_agent.tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    return avg_reward_list


def learning_loop(total_episodes=1000):

    problem = "LunarLanderContinuous-v2"
    env = LunarLanderContinuous()
    # env = SafeLunarEnv(env)

    # if shield_provided != None:
    #     env = SafeLunarEnv(env, shield_provided)
    # else:
    #     env = SafeLunarEnv(env)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    ddpg_agent = DDPG(problem_name=problem,
                      num_states=num_states,
                      num_actions=num_actions,
                      lower_bound=lower_bound,
                      upper_bound=upper_bound,
                      total_episodes=total_episodes)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(ddpg_agent.total_episodes):

        prev_state = env.reset()
        episodic_reward = 0
        render_episodes = 1000
        render = not (ep % render_episodes)

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = ddpg_agent.policy(tf_prev_state, ddpg_agent.ou_noise)
            # Recieve state and reward from environment.
            # print(action)
            state, reward, done, info = env.step(action[0])
            if render: env.render()

            ddpg_agent.buffer.record((prev_state, action[0], reward, state))
            episodic_reward += reward

            ddpg_agent.buffer.learn(ddpg_agent.target_actor,
                                    ddpg_agent.target_critic,
                                    ddpg_agent.actor_model,
                                    ddpg_agent.critic_model,
                                    ddpg_agent.actor_optimizer,
                                    ddpg_agent.critic_optimizer)

            update_target(ddpg_agent.target_actor.variables,
                          ddpg_agent.actor_model.variables, ddpg_agent.tau)
            update_target(ddpg_agent.target_critic.variables,
                          ddpg_agent.critic_model.variables, ddpg_agent.tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    return avg_reward_list


# avg = learning_loop_per_iterations(total_episodes=10)

# avg = learning_loop(total_episodes=1000)
# plt.plot(avg, color='red')
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")

avg = learning_loop(total_episodes=5000)
plt.plot(avg, color='blue')
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")

plt.show()
