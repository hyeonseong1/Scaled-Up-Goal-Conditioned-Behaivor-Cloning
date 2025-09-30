# Scaled Up Goal Conditioned Behavior Cloning! 

This repository is ongoing.
Preparing a paper to submit.

## [Install]    
Recommend python=3.10.x   
```bash
pip install -r requirements.txt
```

## [Command]    
### Run full tests    
```bash
sh run.sh
```

### Run individual test example   
```bash
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py --seed 0
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbcV2.py --seed 0 
```
It takes no more than 1 hour with RTX 3080.  

### Plot performance
```bash
python load_and_plot.py --env medium --algo gcbc
python load_and_plot.py --env medium --algo gcbcV2
```

## [Results]    
Results are saved at eval.csv.    
ex) exp/Debug/antmaze-medium-gcbcV2-sd000/eval.csv


Or see the comprehesive results on wandb. (recommended)

## [Inspiration]
This repository was highly inspired by the following GitHub projects.
1. https://github.com/seohongpark/ogbench
2. https://github.com/dojeon-ai/SimbaV2
