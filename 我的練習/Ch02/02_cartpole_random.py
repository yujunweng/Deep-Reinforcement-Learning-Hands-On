import gym


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
   
    while True:
        action = env.action_space.sample()
        """Returns:
           observation (object): this will be an element of the environment's :attr:`observation_space`.
               This may, for instance, be a numpy array containing the positions and velocities of certain objects.
           reward (float): The amount of reward returned as a result of taking the action.
           terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
               In this case further step() calls could return undefined results.
           truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
               Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
               Can be used to end the episode prematurely before a `terminal state` is reached.
           info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
               This might, for instance, contain: metrics that describe the agent's performance state, variables that are
               hidden from observations, or individual reward terms that are combined to produce the total reward.
               It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
               of returning two booleans, and will be removed in a future version.
           (deprecated)
           done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
               A done signal may be emitted for different reasons: >Maybe the task underlying the environment was solved successfully,
               a certain timelimit was exceeded, or the physics >simulation has entered an invalid state.
        """
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated:
            break
    print("Episode done in %d steps, total reward %.2f"  %(total_steps, total_reward))  