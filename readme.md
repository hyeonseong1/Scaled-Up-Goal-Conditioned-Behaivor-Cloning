### Scaled Up Goal Conditioned Behavior Cloning! 

This repository is ongoing.

[Install]    
Recommend python=3.10.x   
```commandline
pip install -r requirements.txt
```

[Command]    
Train and eval
```commandline
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbc.py --seed 0
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/gcbcV2.py --seed 0 
```

Plot performance
```commandline
python plott.py --env medium --algo gcbc
python plott.py --env medium --algo gcbcV2
```

[Results]    
Results are saved at eval.csv.    
ex) exp/Debug/antmaze-medium-gcbcV2-sd000/eval.csv


Or see the comprehesive results on wandb. (recommended)
