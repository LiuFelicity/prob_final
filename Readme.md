# What we did

### For question 1-1
We change run.py, let it run at least 100 times(most of which run 110 times) with random seed [0,100000] int , and record the seed and iterations.

### For question 1-2
We change run.py and model_multiplier.py.

**run.py**
Let it run at least 100 times(most of which run 110 times) with random weight_seed [0,100000] int, and record the seed and iterations.
We tried 3 different system_init_seed: 62, 300, 1000.

**model_multiplier.py**
Use torch.nn.init.normal_(module.weight, mean=0, std=0.02) to change it from Xavier to Normal.

# What we do to analyze

### For question 1-1
We tried two different ways to calculate p-value, using the formula taught in class and python scripy library.

### For question 1-2
We fisrt set the standard deviation that would make the order matter, and use hypothesis testing to check if data order matters or not.



# 2024 Prob Final Project

### Code Tree Structure
```
prob_final
├── ChickenRabbit.py (making 雞兔同籠 dataset and evaluation method)
├── GCD.py (making gcd dataset and evaluation method)
├── mingpt
│   ├── bpe.py
│   ├── __init__.py
│   ├── model_multiplier.py (define model)
│   ├── trainer_multiplier.py (define trainer)
│   └── utils.py (some helper functions)
├── Readme.md
└── run.py (execute the whole training logic)
```
### Main Purpose
* Using *hypothsis test* and *p-value* to identify the effect of different **model weight initialization** and **data order** on training iterations.
* Project Slide: https://docs.google.com/presentation/d/17T4LfeyejFdhVREXjFXpq5B_bg7Bd9_90NFYxr42mG8/edit?pli=1#slide=id.g2cf94e1e45f_0_0

### Precautions
* You are not allowed to modify the pre-defined model structure (gpt2-mini)
* You only have to modify run.py and mingpt/model_multiplier.py to perform experiments.
* Training can take 20-30 mins per run (on GPU).
