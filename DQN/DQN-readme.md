### DQN implementation notes

# TODO:
First I need the preprocessing function phi, for stacking m frames, gray scaling them, and reshaping them. (done) 1-14-26

Then build out model as described in paper. (done) 1-14-26

Set up the random runs and make sure populating the replay buffer works as expected. (done) 1-16-26

do a test batch gradient desecent on the network using samples from replay buffer, make sure loss and sampling is done correctly.

Then set up dummy training loop with simple hyperparams to make sure it behaves all togeher correctly on a small scale. remember eps annealing. Paper does network SGD update every 4 actions. (done)

Then set up recordings and do a full scale training and hope it works.



# Implementation notes: 

- Gonna set phi_2 = -1 if game terminated in the frames getting to phi_2

- Was going to use deque for buffer but heard access efficiency is bad so I'm sticking with list since I need to randomly sample from buffer

- A lot of the trouble of implementing this is just handling the environment resets and how to store the transitions around the resets.

- I realize now I have no reason to save the actions in the replay buffer and that is wasted computation since I am just inputting the state


# What could I have probably done better:
