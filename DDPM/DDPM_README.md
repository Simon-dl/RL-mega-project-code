# DDPM Implementation


# TODO:

- Get data, just use cifar10 like in old homework. (Done 2-13-26)
https://www.cs.toronto.edu/~kriz/cifar.html


- simple forward pass done (2-13-26)

- Set up U-net model correctly (2-16-26)

- Set up batches and dataloader. (2-16-26)


- Make dummy training loop of predicted added noise and loss on one sample. Set up sampling loop. Make sure diffusion process works on it and you get a solid reconstruction. (can use old homework set up from here to help https://github.com/rll/deepul/blob/master/homeworks/hw4/hw4.ipynb). (2-16-26)


- Have some idea of what "healthy" training should look like before scaling. what nats/dim are you looking for at the end? how should loss lines look? use car image results to get idea (2-16-26)


- Scale it to full dataset, do full training. Start by refactoring the dataset to get all the data. The rest should work with the already batched training done by pytorch.  (2-16-26)

- Sample images in a grid (2-16-26)


# Implementation notes:

I only used the first batch of cifar10 to get some images for the simple training loop set up, during full run should get all images and train on them. Each batch is 10000 images, and each row is a flattened 32x32x3 (3072 dim) image. But I just downloaded an example car image from huggingface so I don't have to unpickle the data each time.


Given U-net archiecture is nice since I don't have to scour the internet for the parameters they used. Still had to get some help understanding it.

Going to make a dataset of just the car image repeating and make test loop with batches since timestep function made for batches.

accidently put model into git, used 'git reset --soft HEAD~1' to rollback then remove model.

Get a semantic check before you do full training, cosine instantly went to 0, and you never used your batch size. Always sanity check.