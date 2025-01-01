# !pip install gym-super-mario-bros==7.3.0
import Mario
import MarioNet
import MetricLogger
import torch
from pathlib import Path
import datetime
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import EnvWrappers
import gym_super_mario_bros

# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", apply_api_compatibility=True)

# Limit the action-space to
#   0. walk right
#   1. jump right
#   2. do nothing
env = JoypadSpace(env, [["right"], ["right", "A"]])
# Apply Wrappers to environment
env = EnvWrappers.SkipFrame(env, skip=4)
env = EnvWrappers.GrayScaleObservation(env)
env = EnvWrappers.ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# パイパーパラメータ群
EPISODES = 1000

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

train = True
which_network = str(1)

if train:

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario.Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger.MetricLogger(save_dir)

    for e in range(EPISODES):

        state, info = env.reset()
        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, truncated, info = env.step(action)

            # env.render()

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
else:
    load_path = "checkpoints/2021-07-18T19-05-02/mario_net_" + which_network + ".chkpt"
    checkpoint = torch.load(load_path)
    network = MarioNet.MarioNet(input_dim=(4, 84, 84), output_dim=env.action_space.n)
    network.load_state_dict(checkpoint['model'])
    while True:
        state = env.reset()
        while True:
            state = state.__array__()
            state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = network(state, 'online')
            action = torch.argmax(action_values, axis=1).item()
            next_state_, reward, done, info = env.step(action)
            state = next_state_
            env.render()
            if done or info["flag_get"]:
                break