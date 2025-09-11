# Scaled Up Goal Conditioned Behavior Cloning! 

This repository is ongoing.

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

### Run individual tests example   
```bash
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py --seed 0
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbcV2.py --seed 0 
```
It takes no more than 1 hour with RTX 3080.  

### Plot performance
```bash
python plott.py --env medium --algo gcbc
python plott.py --env medium --algo gcbcV2
```

## [Results]    
Results are saved at eval.csv.    
ex) exp/Debug/antmaze-medium-gcbcV2-sd000/eval.csv


Or see the comprehesive results on wandb. (recommended)
