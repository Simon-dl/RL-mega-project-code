# Diffusion policy implemention


This works with low_dim datasets from robomimic:

https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

Create a dataset folder and put the dataset in it. Then in main just change the datset variable and it will run.

Depending on what dataset you use the state variable dim will be different, the dataset function should tell you which dimensions it is.
You will need to update the model obs_dim to match

if you want to load and run your model on multiple evaluations uncomment the code in eval_and_viz.py