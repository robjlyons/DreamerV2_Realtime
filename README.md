# An Implementation Of DreamerV2 On Real Games

## Step 1 - Install Custom Env

### Requirements

For windows only, other OS might just require some googling.

I strongly suggest setting up a conda env for this - [Creating Conda Envs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

```
pip install opencv-python numpy matplotlib gym pytessy pywin32
```

Pytessy is used as an alternative to pytesseract for its speed whilst sacrificing accuracy. If you have issues installing pytessy then please let me know and I will try and help, otherwise try pytesseract instead.

### Setting up in Gym

I then manually add the environment to gym

- navigate to your conda environment - lib\site-packages\gym\envs\classic_control folder
- Place retroarch.py here (from the env folder)
- Open the init.py file in classic_control
- Add the following line at the bottom
```
from gym.envs.classic_control.retroarch import RetroArch
```
- Go up to the gym\envs folder
- Open the init.py file in envs
- Add the following code at the bottom of the Classic section underneath Acrobat
```
register(
    id='RetroArch-v0',
    entry_point='gym.envs.classic_control:RetroArch',
)
```
This allows dreamer to recognise the custom environment. It will be used as an atari env but I have placed it in classic control for 2 reasons, 1. the atari envs are registered differently and it will error if placed in the folder. 2. the atari env folder doesn't exist in newer gym versions so this allows the custom env to be used in newer versions of gym/gymnasium.

## Step 2 - Install DreamerV2

You can install dreamerV2 using the instructions in their [repository](https://github.com/danijar/dreamerv2)

## Step 3 - Run Dreamer On Your Custom Env

you can make any edits you want to the dream.py and/or the dreamerv2/configs.yaml files and then run:

```
python dream.py
```

## Basic Troubleshooting

First things first, I am running this exact configuration on a Windows machine (AMD Ryzen 3700X, Nvidia RTX3070ti, 32gb DDR4)

your mileage WILL vary depending on your specs. If you have less RAM then I would suggest lowering the replay capacity in configs.yaml (replay: {capacity: 1.5e6, ongoing: False, minlen: 50, maxlen: 50, prioritize_ends: True}).

The algorithm will not learn if the custom env is not frame throttled. I have not tested throttling to 8 fps and then doubling the emulator speed. This WILL require better hardware to run for long periods.

## Other Games

If you'd like to run this on other games then please refer to the retroarch.py file. I have documented the file with instructions on how to do this.
