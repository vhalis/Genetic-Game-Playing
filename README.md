# 2048
Playing 2048 with genetically selected neural nets

```
from trainer import Genetic2048Trainer as GT
t = GT(iterations_per_model=300, experiment_name='largenet_weightedbreeding', debug=True, score_algorithm=lambda x: x.valid_moves,
       breeding_algorithm='score_weighted', hidden_sizes=(16,8), generations=1000, generation_size=1000, invalid_move_ends_game=True)
w_out = t.run_experiment()
```
