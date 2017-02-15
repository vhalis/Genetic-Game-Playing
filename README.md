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
w_out = t.resume_experiment(epoch_num=45)
```
