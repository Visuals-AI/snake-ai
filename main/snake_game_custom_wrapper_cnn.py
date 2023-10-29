import math

import gym
import numpy as np

from snake_game import SnakeGame



# 游戏环境，SnakeEnv 设计成符合 OpenAI Gym 的环境规范的
# OpenAI Gym 提供了一个标准的环境接口，使得研究人员和开发人员可以在不同的强化学习算法上使用相同的环境，
# 而无需对代码进行大量修改。OpenAI Gym 的环境规范主要包括以下几个核心方法：
#   1. __init__(): 构造函数，用于初始化环境的状态和参数。
#   2. reset(): 重置环境到一个初始状态并返回初始观测。
#   3. step(action): 根据智能体的动作更新环境状态，并返回新的观测、奖励、是否完成以及其他有关信息。
#   4. render(): 可选的，用于显示或渲染环境的当前状态，通常用于调试或可视化。
# SnakeEnv 实现了这些核心方法，所以它是与 Gym 兼容的。
# 这意味着你可以在任何支持 Gym 环境的强化学习算法上使用它。
class SnakeEnv(gym.Env):

    # 用于初始化环境的状态和参数。
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        # 表示一个离散的动作空间，其中动作是从0开始的整数。
        # 在这种情况下，gym.spaces.Discrete(4)表示有四个动作：0、1、2、3。这本身并没有为这些整数提供语义或描述。
        # 换句话说，Gym并不关心每个动作的“实际”含义或名称。它只需要知道有多少个可能的动作和它们的范围。
        # 这使得定义动作空间变得非常通用和灵活，因为你可以在实际的step方法中为每个数字定义具体的动作语义。
        # 这种设计的目的是为了使环境定义尽可能地通用和简洁。
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        # 对于更复杂的动作空间，例如在网游中同时处理移动和释放技能，你可以使用更复杂的动作空间表示方法。
        # OpenAI Gym提供了多种动作空间类型，其中一种是spaces.Tuple，允许你组合多个动作空间。
        # 甚至如果你的动作空间是连续的，例如一个机器人的关节角度或加速器的压力，
        # 那么你可能会使用 spaces.Box 来表示这个空间。
        
        
        # 观察值的格式或结构应与observation_space的声明相一致。
        # observation_space 被定义为一个维度为 (84, 84, 3) 的框 (gym.spaces.Box)，其中每个元素的值都在0到255之间。
        # 因此，_generate_observation 返回的观察值应该满足这些约束。
        # 如对于这种图像形式的观察值，强化学习算法常常使用卷积神经网络 (Convolutional Neural Networks, CNN)。CNN 能够从图像中提取特征，并根据这些特征来决策。
        # Gym提供了一个框架，observation_space 就是声明了一个统一的接口来与环境互动。实际上，如何处理和解释这些观察值是强化学习模型的任务。
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    # 重置环境到一个初始状态并返回初始观测。
    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs
    
    # 根据智能体的动作更新环境状态，并返回新的观测、奖励、是否完成以及其他有关信息。
    # 返回值是 obs, reward, done, info
    #   obs: 产生当前环境状态的观察值，代表了环境在执行动作之后的新状态。在很多游戏和任务中，观察值 (observation) 为智能体提供了关于环境当前状态的必要信息，以便它能做出下一个决策。在 SnakeEnv 的上下文中，obs 可能包含了游戏板上的蛇、食物和其他可能的障碍物的当前位置。对于深度学习和强化学习模型，这些观察值会被用作输入来决策下一步的动作。
    #   reward:
    #   done:
    #   info: 是一个自定义字典（或称为哈希），它提供了关于当前步骤或环境的额外信息。Gym不会直接使用info字典来“学习”。换句话说，标准的强化学习算法在更新策略或学习模型时并不依赖于info。然而，info可以在训练时为研究人员或开发者提供有用的上下文或调试信息。
    def step(self, action):
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        # 胜利奖励：
        # 当蛇的大小达到了整个棋盘的大小（即填满了整个棋盘），蛇将获得胜利奖励，这个奖励是其最大增长的0.1倍
        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
        
        # 游戏结束的惩罚：
        # 如果蛇碰到墙或自己，游戏结束，这时，奖励会是负数，并根据蛇的大小和最大可能增长来计算。
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            reward = reward * 0.1
            return obs, reward, self.done, info
          
        # 食物奖励：
        # 当蛇吃到食物时，奖励将基于蛇的当前大小与棋盘大小的比例。
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0 # Reset reward step counter
        
        else:

            # 方向奖励/惩罚：
            # 除了明显的奖励和惩罚，蛇还会根据其是否朝向食物获得一个微小的奖励或惩罚。如果蛇在最新的步骤中更接近食物，它会获得一个正奖励，反之则会受到一个小的惩罚
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward, self.done, info
    
    # 可选的，用于显示或渲染环境的当前状态，通常用于调试或可视化。
    def render(self):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction # 当前蛇的移动方向
        snake_list = self.game.snake            # 当前蛇的身体位置
        row, col = snake_list[0]

        # 根据传入的动作，此方法检查该动作是否与当前方向相反
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # 检查执行该动作后蛇的头是否会碰到墙壁或蛇自己的身体
        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # 返回一个维度为 (84, 84, 3) 的 NumPy 数组。这意味着它是一个84x84的图像，并有三个颜色通道（RGB）。
    # 因此，这个观察值实际上是一个图像，它展示了游戏板上的蛇、食物和其他物体的位置。
    # 观察值的格式或结构应与observation_space的声明相一致。
    # observation_space 被定义为一个维度为 (84, 84, 3) 的框 (gym.spaces.Box)，其中每个元素的值都在0到255之间。
    # 因此，_generate_observation 返回的观察值应该满足这些约束。
    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
