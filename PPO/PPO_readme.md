### DQN implementation notes

# TODO:

    -   Get resources on how to set up multiple environments with multiple agents and parallel. 
    Going to start with a single agent in one with Hopper https://gymnasium.farama.org/environments/mujoco/hopper/ to make sure I understand the basics

    and then extend to synchronous agents in Walker2d later https://gymnasium.farama.org/environments/mujoco/walker2d/ for full training.
 
    https://gymnasium.farama.org/api/vector/sync_vector_env/
    (Done) 1-21-26

    - Start with setting up model for a single env and agent in hopper, make sure it can properly select actions from a continous distribution and they are in the expected bounds. (done) 1-23-26

    - Set up trajectory saver and value network. (Done) 1-26-26

    - Set up PPO loss function, with seperate function for calculating advantage, remember multiple epochs and mini-batch size. (done) 1 -27-26

    - Figure out value function loss. (done) 1 -27-26

    - Do simple training run with one agent. Make sure reward is going up and value loss going down. (done) 1 -27-26

    - Extend to multiple agents running in parallel and update loss and environments accordingly. (done) 1 -28-26

    - Add CUDA and video recording (done) 1-28-26




    


# Implementation notes:

    - Looking at the environment for hopper, despite how the actions are now continous the state is only 17 values. Compared to the 210x160x3 for a single state of Atari (not even including all the different 255 values each pixel could take on in each channel), it is very very simple. I was confused on why the model was so small in the paper but now it makes sense, anything more would be overkill.

    - I will need to look up how to output the mean with variable standard deviation since I didn't read the TRPO or benchmarking paper where that is explained (Or I assume explained). 

    -Remember only numerator policy and predicted state value has gradient, I think GAE with batches is going to be the biggest headache of this project. 

    -Paper doesn't really talk about value network training for GAE calculation, I assume it's one of those "people should know this so we are not including it in the paper" sort of things.

    -Now I need to extend to multi-actor, it looks like each actor gets a trakectory in their own environment using the current policy, then calculates their own advantage estimates for their trajectories. Then since those are fixed we can stack the horizons and trajectories into a dataset and sample from them? Ai says it's correct but that I need to normalize advantages from combined dataset. 

    -If I really wanted to parallel process I should make it so all the advanatges calculated in synch using that function, I might as well try that.

    -Need to figure out how to gracefully reset sub environments so one can reset while the other still goes. Thankfully neural nets have no problem working with batches. After that to do the advantage function need to find way to filter through collected data so each are in their right stream.

    -Once again back to fiddling with shapes to make sure everything is correct.