
import gymnasium as gym
import ale_py #needed for namespace
import cv2
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from DQN import phi
from DQN import DQN
import torch
import random
import time
import tqdm

def populate_buffer(env,replay_buffer,frame_skip,amount_to_pop):
    """
    Takes in env, buffer, frames to skip, and amount to populate (in number of frames). 

    returns: total_frame_count

    populates replay buffer with some transition dynamics
    """
    phi_1 = 0
    phi_2 = 0

    #phi_1 = torch.tensor(phi(frames),dtype=torch.float)

    total_frame_count = 0
    #populate replay buffer with random plays, split into helper function later
    frames = []
    observation, _ = env.reset()
    frames.append(observation)
    total_frame_count += 1
    old_action = 1
    old_reward = 0

    action = 1 #start with fire

    for i in range(3): #setting up
        total_frame_count += 1
        observation, reward, terminated, truncated, _ = env.step(action)
        frames.append(observation)
        if len(frames) == frame_skip:
            action = env.action_space.sample() 
            phi_1 = phi(frames)
            phi_2 = phi_1

            old_action = action
            old_reward = reward
            frames = []

    while total_frame_count <= amount_to_pop:
        observation, reward, terminated, truncated, _ = env.step(action)
        frames.append(observation)
        total_frame_count += 1

        if terminated or truncated:
            phi_1 = phi_2
            phi_2 = -1
            transition = (phi_1,old_action,old_reward,phi_2)
            replay_buffer.append(transition)

            frames = []
            observation, _ = env.reset()
            frames.append(observation)
            action = env.action_space.sample() 
            old_action = action
            old_reward = 0


        if len(frames) == frame_skip:
            # Check if this is the first state after a reset (phi_2 is -1)
            if not isinstance(phi_2, np.ndarray) and phi_2 == -1:
                phi_1 = phi(frames)
                phi_2 = phi_1
                frames = []
                continue
            else:
                phi_1 = phi_2
                phi_2 = phi(frames) 
            transition = (phi_1,old_action,old_reward,phi_2)
            replay_buffer.append(transition)

            action =  env.action_space.sample() 
            old_action = action
            old_reward = reward
            frames = []
    

    return  total_frame_count 

def eps_anneal(initial, final, total_frames):
    """
    Returns a function that calculates epsilon for a given frame.
    """
    step_size = (initial - final) / total_frames
    
    def get_epsilon(current_frame):
        epsilon = initial - step_size * current_frame
        return max(final, min(initial, epsilon))
    
    return get_epsilon

def eval_model(model,frame_skip,eval_num):
    env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger = lambda num: num % 1 == 0,
        video_folder="saved-video-folder",
        name_prefix=f"video-{eval_num}",
    )

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_over = False

    frames = []
    observation, _ = env.reset()
    frames.append(observation)

    action = 1
    total_reward = 0
    while not episode_over:
            observation, reward, terminated, truncated, _ = env.step(action)
            frames.append(observation)

            #just greedy action selection for eval
            if len(frames) == frame_skip:
                processed_frames = torch.tensor(phi(frames),dtype=torch.float).to(device)
                action = torch.argmax(model(processed_frames)).item()
                frames = []

            total_reward += reward
            episode_over = terminated or truncated

    end = time.time()
    print(f"Eval took {end - start} seconds")

    env.close()

    return total_reward



def breakout_training():
    #render_mode="human" for when I want to watch an episode

    env = gym.make('ALE/Breakout-v5')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    max_buffer_size = 30000
    replay_buffer = []
    total_frame_count = 0 

    behavior_model = DQN(4).to(device) #4 is output actions
    target_model = DQN(4).to(device)

    lr = 0.00025
    optimizer = torch.optim.Adam(behavior_model.parameters(),lr=lr)
    MSE_loss = torch.nn.MSELoss()


    #target network updare frequence.
    network_update_freq = 5000

    #Do SGD updates after this many actions
    update_frequency = 4 
    action_count = 0

    #other hyper params
    episodes = 10000
    discount = .99
    total_frame_count = 0 
    batch_size = 32

    #eps annealing hardcode, for 100k frames rather than 1M like in paper
    eps_val = eps_anneal(1,.1,1000000)

    #eval
    do_eval = False
    eval_step = 250000
    eval_rewards = []
    eval_num = 0

    #actions
    no_op_action = 0
    
    episode_rewards = []

    frame_skip = 4

    phi_1 = 0
    phi_2 = 0

    pop_frame_count = 25000

    total_frame_count = populate_buffer(env,replay_buffer,frame_skip,pop_frame_count) 
    avg_time = []

    

    print("starting episodes")

    #inital eval
    # eval_num += 1
    # eval_rewards.append(eval_model(behavior_model,frame_skip,eval_num)) 

    for i in tqdm.tqdm(range(episodes)): 
        start = time.time()
        episode_over = False

        frames = []
        observation, _ = env.reset()
        frames.append(observation)
        total_frame_count += 1

        phi_2 = -1 #make sure 
        action = 1
        old_action = 1
        old_reward = 0
        total_reward = 0

        while not episode_over:
            observation, reward, terminated, truncated, _ = env.step(action)
            total_frame_count += 1
            frames.append(observation)


            if total_frame_count % eval_step == 0:
                print("here")
                do_eval = True

            if len(frames)== frame_skip:
                if not isinstance(phi_2, np.ndarray) and phi_2 == -1:
                    phi_1 = phi(frames)
                    phi_2 = phi_1
                    frames = []
                    continue
                else:
                    phi_1 = phi_2
                    phi_2 = phi(frames) 
                transition = (phi_1,old_action,old_reward,phi_2)
                replay_buffer.append(transition)


                #select e-greedy action
                if np.random.random() < eps_val(total_frame_count):
                    action = env.action_space.sample()
                else:
                    processed_frames = torch.tensor(phi_2,dtype=torch.float).to(device)
                    action = torch.argmax(behavior_model(processed_frames)).item()


                #check if it's SGD time
                if action_count == update_frequency:
                    optimizer.zero_grad()
                    action_count = 0


                    # minibatch = random.choices(replay_buffer, k=batch_size) # with replacement
                    minibatch = random.sample(replay_buffer, min(batch_size, len(replay_buffer))) #without

                    # Batch all states and next_states
                    states = []
                    next_states = []
                    actions = []
                    rewards = []
                    dones = []

                    for transition in minibatch:
                        states.append(transition[0])
                        actions.append(transition[1])
                        rewards.append(transition[2])
                        next_state = transition[3]
                        if isinstance(next_state, np.ndarray):
                            next_states.append(next_state)
                            dones.append(False)
                        else:
                            next_states.append(transition[0])  # dummy state for terminal
                            dones.append(True)

                    # Convert to batched tensors
                    states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(device).squeeze(1)
                    next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(device).squeeze(1)
                    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
                    dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device)

                    # Batch forward passes
                    with torch.no_grad():
                        next_q_values = target_model(next_states_tensor)
                        next_q_max = torch.max(next_q_values, dim=1)[0]
                        target_Qs = rewards_tensor + (~dones_tensor).float() * discount * next_q_max

                    current_q_values = behavior_model(states_tensor)
                    pred_Q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                    loss = MSE_loss(pred_Q, target_Qs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(behavior_model.parameters(), max_norm=10)
                    optimizer.step()


                old_action = action
                old_reward = reward
                frames = []
                action_count += 1
                

            
            #update target model every C frames
            if total_frame_count % network_update_freq == 0:
                target_model.load_state_dict(behavior_model.state_dict())

            total_reward += reward
            episode_over = terminated or truncated
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

        episode_rewards.append(total_reward)
        end = time.time()
        avg_time.append(end - start)

        if do_eval == True: # do at end of episode to not mess up current episode 

            print("eval",eval_num, "frame", total_frame_count)
            eval_num += 1
            eval_rewards.append(eval_model(behavior_model,frame_skip,eval_num))
            do_eval = False
            
    print(eval_num)
    eval_num += 1
    print(eval_num)
    eval_rewards.append(eval_model(behavior_model,frame_skip,eval_num)) #one final eval
    print(len(replay_buffer))
    print("final frame count", total_frame_count - pop_frame_count)
    env.close()
    return episode_rewards, eval_rewards, avg_time


reward_list, eval_rewards, time = breakout_training()
print(reward_list[-10:])
print(eval_rewards)
print(sum(time)/len(time))



