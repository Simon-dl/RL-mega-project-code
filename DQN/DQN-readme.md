### DQN implementation notes

TODO:
First I need the preprocessing function phi, for stacking m frames, gray scaling them, and reshaping them. (done) 1-14-26

Then build out model as described in paper. (done) 1-14-26

Set up the random runs and make sure populating the replay buffer works as expected. remember eps annealing

do a test batch gradient desecent on the network using samples from replay buffer, make sure loss and sampling is done correctly.

Then set up dummy training loop with simple hyperparams to make sure it behaves all togeher correctly on a small scale.
Paper does network SGD update every 4 actions.

Then set up recordings and do a full scale training and hope it works.