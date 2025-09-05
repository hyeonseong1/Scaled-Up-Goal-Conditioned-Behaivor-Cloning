### Scaled Up Goal Conditioned Behavior Cloning! 

This repository is ongoing.
[Install]
```commandline
pip install -e requirements.txt
```

[command]

train and eval
```commandline
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episode=50 --agent=agents/gcbc.py --seed 0
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episode=50 --agent=agents/gcbcV2.py --seed 0 
```

plot performance
```commandline
python plott.py --env medium --algo gcbc
python plott.py --env medium --algo gcbcV2
```
