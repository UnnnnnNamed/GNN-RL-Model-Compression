import torch.backends.cudnn as cudnn
import torch
import logging
import numpy as np
import sys
import os

# 获取当前文件的上级目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将上级目录路径添加到 sys.path
sys.path.append(parent_dir)
from gnnrl.lib.RL.agent import Memory
logging.disable(30)

torch.backends.cudnn.deterministic = True
def transfer_policy_search():
    return

def search(env,agent, update_timestep,max_timesteps, max_episodes,
           log_interval=10, solved_reward=None, random_seed=None):
    print("Search Start")
    ############## Hyperparameters ##############
    print('Hyperparameters')
    env_name = "gnnrl_search"
    render = False
    solved_reward = solved_reward  # stop training if avg_reward > solved_reward
    log_interval = log_interval  # print avg reward in the interval
    max_episodes = max_episodes  # max training episodes
    max_timesteps = max_timesteps  # max timesteps in one episode

    update_timestep = update_timestep  # update policy every n timesteps

    random_seed = random_seed
    #############################################


    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()



    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    print("-*" * 10, "start search the pruning policies", "-*" * 10)
    # training loop
    # 在每个episode结束时重置环境
    # 训练循环中增加中间奖励机制
    # avg_reward_window = [] # 新增滑动窗口记录奖励
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()  # 确保每次episode开始前重置环境
        t=0
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = agent.select_action(state, memory)
            state, reward, done = env.step(action, t + 1)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                print("-*" * 10, "start training the RL agent", "-*" * 10)
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
                print("-*" * 10, "start search the pruning policies", "-*" * 10)

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if (i_episode % log_interval) != 0 and running_reward / (i_episode % log_interval) > (solved_reward):
            print("########## Solved! ##########")
            torch.save(agent.policy.state_dict(), './rl_solved_{}.pth'.format(env_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(agent.policy.state_dict(), './'  + '_rl_{}.pth'.format(env_name))
            torch.save(agent.policy.actor.state_dict(), './'+'_rl_actor_{}.pth'.format(env_name))
            torch.save(agent.policy.critic.state_dict(), './'+'_rl_critic_{}.pth'.format(env_name))
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        # # 新增中间奖励机制（每10步记录一次）
        # if i_episode % log_interval == 0:
        #     avg_reward_window.append(running_reward / log_interval)
        #     if len(avg_reward_window) > 20 and all(r < -70 for r in avg_reward_window[-5:]): # 若持续表现差则重置
        #         env.reset()
        #         agent.policy.load_state_dict(agent.policy_old.state_dict())
        #         avg_reward_window = []
    print('SEARCH END')
