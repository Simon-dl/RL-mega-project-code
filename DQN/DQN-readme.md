### DQN implementation notes

# TODO:
First I need the preprocessing function phi, for stacking m frames, gray scaling them, and reshaping them. (done) 1-14-26

Then build out model as described in paper. (done) 1-14-26

Set up the random runs and make sure populating the replay buffer works as expected. (done) 1-16-26

do a test batch gradient desecent on the network using samples from replay buffer, make sure loss and sampling is done correctly. (done) 1-17-26

Then set up dummy training loop with simple hyperparams to make sure it behaves all togeher correctly on a small scale. remember eps annealing. Paper does network SGD update every 4 actions. (done)

Add evaluation every 250000 frames

Then set up recordings and do a full scale training and hope it works.



# Implementation notes: 

- Gonna set phi_2 = -1 if game terminated in the frames getting to phi_2

- Was going to use deque for buffer but heard access efficiency is bad so I'm sticking with list since I need to randomly sample from buffer

- A lot of the trouble of implementing this is just handling the environment resets and how to store the transitions around the resets.

- I realize now I have no reason to save the actions in the replay buffer and that is wasted computation since I am just inputting the state

- Pre-Cuda usually around 2 seconds per episode, after cuda around .5 seconds per episode.
- After more batching improvements .25 per episode. A full 800 run usually takes around 4 minutes for me, but ideally it will take longer as the agent stays alive longer

-Relooking at the paper I did not realize the agents were trained for 10 million frames, that's crazy and I have to update my episode count by a lot. 800 episodes gets me 280,000 frames (not including inital 25000 frames), no wonder they annealed eps over 1000000. 


# What could I have probably done better:
