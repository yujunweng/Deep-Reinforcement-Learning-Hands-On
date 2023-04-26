from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import gymnasium as gym



if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    before_training = r"./videos\before_training.mp4"
    video = VideoRecorder(env, before_training)
    
    env.reset()
    total_reward = 0.0
    total_steps = 0
    
    while True:
        env.render()    # 視覺化呈現，它只會回應出呼叫那一刻的畫面給你，要它持續出現，需要寫個迴圈。
        video.capture_frame()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break
            
    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
    video.close()
    env.close()  