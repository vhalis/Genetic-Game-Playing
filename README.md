# 2048
Playing 2048 with genetically selected neural nets.
Google Slides presentation [here](https://docs.google.com/presentation/d/1C7uS-Fwj0bVExR3NH2OK9Ap3zBWG5yf6gn_k8SW6NvY/edit?usp=sharing).

```
from trainer import Genetic2048Trainer as GT
t = GT(iterations_per_model=300, experiment_name='largenet_weightedbreeding',
       debug=True, score_algorithm=lambda x: x.valid_moves,
       breeding_algorithm='score_weighted', hidden_sizes=(16,8),
       generations=1000, generation_size=1000, invalid_move_ends_game=True)
w_out = t.run_experiment()
```

# Tic Tac Toe

To start a new experiment:
```
from trainer import TicTacToeTrainer as TT
t = TT(iterations_per_model=20, experiment_name='ttt_1',
       debug=True, score_algorithm=lambda x: x.valid_moves * x.game_score,
       breeding_algorithm='uniform',
       hidden_sizes=(4,), inputs=9, outputs=9,
       generations=1000, generation_size=1000, generation_cutoff=0.3,
       invalid_move_ends_game=True)
w_out = t.run_experiment()
```

To resume an old experiment:
```
from trainer import TicTacToeTrainer as TT
# Use the same parameters for the trainer as when you started the experiment
# To be fixed later
t = TT(iterations_per_model=20, experiment_name='ttt_1',
       debug=True, score_algorithm=lambda x: x.valid_moves * x.game_score,
       breeding_algorithm='uniform',
       hidden_sizes=(4,), inputs=9, outputs=9,
       generations=1000, generation_size=1000, generation_cutoff=0.3,
       invalid_move_ends_game=True)
# If you don’t want to overwrite epochs, set the epoch num on the trainer
t.epoch_num=45
w_out = t.resume_experiment()
# If you don’t care, don’t set it on the trainer and load the epoch directly
# For example if the old experiment name differs
w_out = t.resume_experiment(epoch_num=45, experiment_name='ttt_2')
```

To get stats for an experiment, you can specify an epoch or get all stats at once:
```
t.experiment_stats()
```
This will place a `stats.csv` file in the experiments folder of the experiment name.
Accepted parameters are:
* epoch_start - For the epoch number to start from
* experiment_name - If the name of the experiment is different than the experiment name of the trainer
* redo_epochs - If previous stats made should be overwritten
