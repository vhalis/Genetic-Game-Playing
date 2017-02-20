import math
import os
import random

from operator import (
    __add__,
    attrgetter,
    )
from collections import namedtuple

import numpy

from game import (
    Board as Board2048,
    GameOverOverflow,
    )
from nboard import (
    GameOver,
    )
from nets import Net
from ttt import (
    BoardTTT,
    GameOverInvalid,
    NaiveTTTPlayer,
    )


ModelStats = namedtuple('ModelStats',
                        ['weights_idx',
                         'score',
                         'score_parameters',
                         ])


ModelScore = namedtuple('ModelScore',
                        ['game_over',
                         'game_score',
                         'invalid_moves',
                         'valid_moves',
                         ])


class GeneticNetTrainer(object):

    DEFAULT_BREEDING_ALGORITHM = 'uniform'
    DEFAULT_EPOCH_NUM = 0
    DEFAULT_EPOCH_SIZE = 5
    DEFAULT_EXPERIMENT_NAME = 'Default'
    DEFAULT_GENERATIONS = 100
    DEFAULT_GENERATION_CUTOFF = 0.5
    DEFAULT_GENERATION_SIZE = 1000
    DEFAULT_MUTATION_CHANCE = 0.1
    DEFAULT_MUTATION_DIFFERENCE = 0.05
    DEFAULT_POLYGAMY = False

    DEFAULT_MODEL_ITERATIONS = 1000

    BREEDING_ALGORITHM_CHOICES = [
        'score_weighted',
        'uniform',
        ]

    def __init__(self,
                 # Genetic parameters
                 breeding_algorithm=DEFAULT_BREEDING_ALGORITHM,
                 generations=DEFAULT_GENERATIONS,
                 generation_cutoff=DEFAULT_GENERATION_CUTOFF,
                 generation_size=DEFAULT_GENERATION_SIZE,
                 mutation_chance=DEFAULT_MUTATION_CHANCE,
                 mutation_difference=DEFAULT_MUTATION_DIFFERENCE,
                 polygamous=DEFAULT_POLYGAMY,
                 # Model parameters
                 iterations_per_model=DEFAULT_MODEL_ITERATIONS,
                 # Epoch parameters
                 epoch_num=DEFAULT_EPOCH_NUM,
                 epoch_size=DEFAULT_EPOCH_SIZE,
                 experiment_name=DEFAULT_EXPERIMENT_NAME,
                 **kwargs):
        if (not isinstance(generations, int) or
                not isinstance(generation_size, int) or
                generations < 1 or generation_size < 1):
            raise ValueError('Generations and generation size must be positive'
                             ' integers greater than zero')
        if (not isinstance(generation_cutoff, float) or
                generation_cutoff <= 0.0 or generation_cutoff >= 1.0):
            raise ValueError('Generation cutoff must be a positive float '
                             ' between 0.0 and 1.0 (exclusive)')

        if (not isinstance(breeding_algorithm, basestring) or
                breeding_algorithm not in self.BREEDING_ALGORITHM_CHOICES):
            raise ValueError('Unknown net breeding algorithm: {}'.format(
                breeding_algorithm))
        if (not isinstance(mutation_chance, float) or
                not isinstance(mutation_difference, float) or
                mutation_chance < 0.0 or mutation_chance > 1.0 or
                mutation_difference < 0.0 or mutation_difference > 1.0):
            raise ValueError('Mutation chance, and mutation difference must be'
                             ' positive floats between 0.0 and 1.0'
                             ' (inclusive)')

        if not isinstance(polygamous, bool):
            raise ValueError('Polygamous must be true or false')

        if (not isinstance(iterations_per_model, int) or
                iterations_per_model <= 0):
            raise ValueError('Number of iterations per model must be a positive'
                             ' integer')

        if (not isinstance(epoch_size, int) or not isinstance(epoch_num, int)
                or epoch_size <= 0 or epoch_num < 0):
            raise ValueError('Epoch size must be an integer greater than zero'
                             ' and epoch num must be a non-negative integer')
        if not experiment_name or not isinstance(experiment_name, basestring):
            raise ValueError('Epoch name must be a non-empty string')

        # Nets
        net_hidden_sizes = kwargs.pop('hidden_sizes', Net.DEFAULT_HIDDEN_SIZES)
        net_weights = kwargs.pop('weights', Net.DEFAULT_WEIGHTS)
        net_inputs = kwargs.pop('inputs', Net.DEFAULT_INPUTS)
        net_outputs = kwargs.pop('outputs', Net.DEFAULT_OUTPUTS)
        net_spread = kwargs.pop('weight_spread', Net.DEFAULT_WEIGHT_SPREAD)
        net_middle = kwargs.pop('weight_middle', Net.DEFAULT_WEIGHT_MIDDLE)
        # Test to ensure nothing breaks
        Net(hidden_sizes=net_hidden_sizes, weights=net_weights,
            inputs=net_inputs, outputs=net_outputs,
            weight_spread=net_spread, weight_middle=net_middle)

        # Debug variables
        self.debug = kwargs.pop('debug', None)
        self.get_model_score = kwargs.pop('score_algorithm',
                                          self.get_model_score)
        # If kwargs not empty, extra key words were passed
        if len(kwargs.keys()) > 0:
            raise ValueError('Unnecessary kwargs passed to GeneticNetTrainer:'
                             ' {}'.format(','.join(kwargs.keys())))

        # Genetic variables
        self.breeding_algorithm = breeding_algorithm
        self.generations = generations
        self.generation_size = generation_size
        self.generation_cutoff = generation_cutoff
        self.mutation_chance = mutation_chance
        self.mutation_difference = mutation_difference
        self.polygamous = polygamous
        # Model variables
        self.iterations_per_model = iterations_per_model
        # Net variables
        self.net_hidden_sizes = net_hidden_sizes
        self.net_weights = net_weights
        self.net_inputs = net_inputs
        self.net_outputs = net_outputs
        self.net_spread = net_spread
        self.net_middle = net_middle
        # Epoch variables
        self.epoch_size = epoch_size
        self.epoch_num = epoch_num
        self.experiment_name = experiment_name

    def breed_new_generation(self, weights, scores=None):
        # Weights is a list of lists of numpy.ndarrays
        # Breed generation in a 'seed' competition format
        seeds = len(weights)
        partner_offset = -1 if self.polygamous else 0
        polygamy_offset = 0 if not self.polygamous else 1
        next_gen = [None for _ in xrange(self.generation_size)]
        for offspring_num in xrange(self.generation_size):
            idx = offspring_num % seeds
            if idx == 0:
                partner_offset += polygamy_offset
            pair_idx = (seeds - offspring_num - partner_offset) % seeds
            if pair_idx == idx:
                # Don't breed with self - use highest seed instead
                pair_idx = 0 if idx != 0 else 1
            if scores:
                next_gen[offspring_num] = self.breed_organisms(
                    weights[idx], weights[pair_idx],
                    scores[idx], scores[pair_idx]
                    )
            else:
                next_gen[offspring_num] = self.breed_organisms(
                    weights[idx], weights[pair_idx],
                    )
        return next_gen

    def breed_organisms(self, weights_1, weights_2,
                        scores_1=None, scores_2=None):
        """
        Breed two organisms (multiple matrices each) using 'uniform crossover'
        All weights in weights_1 and weights_2 must have the same shape & dtype
        and weights_1 and weights_2 must have the same length
        """
        w_out = [None for _ in xrange(len(weights_1))]
        for idx, (w_1, w_2) in enumerate(zip(weights_1, weights_2)):
            w_out[idx] = self.breed_pair(w_1, w_2, scores_1, scores_2)
        return w_out

    def breed_pair(self, weights_1, weights_2, score_1=None, score_2=None):
        """
        Breed pair using a 'uniform crossover' algorithm by default
        weights_1 and weights_2 must have the same shape & dtype
        """
        w_shape = weights_1.shape
        w_dtype = weights_1.dtype
        weights_1 = weights_1.flatten()
        weights_2 = weights_2.flatten()
        weights_out = numpy.zeros(w_shape, dtype=w_dtype)
        for idx, (w_1, w_2) in enumerate(zip(weights_1, weights_2)):
            if self.breeding_algorithm == 'uniform':
                w_temp = w_1 if random.random() < 0.5 else w_2
            elif self.breeding_algorithm == 'score_weighted':
                if not score_2:
                    score_2 = 1.0
                # Threshold becomes larger when score_1 is larger than score_2
                threshold = 1 / (1 + math.exp((score_2 - score_1) /
                                              abs(float(score_2))))
                w_temp = (w_1 if random.random() < threshold else w_2)
            else:
                raise ValueError('Unknown breeding algorithm selected:'
                                 ' {}'.format(self.breeding_algorithm))
            if random.random() > 1.0 - self.mutation_chance:
                updown = [1, -1][random.randint(0, 1)]
                w_temp = w_temp + (w_temp * updown * self.mutation_difference)
            weights_out.put(idx, w_temp)

        return weights_out

    def do_epoch(self, weights):
        self.save_generation(weights)
        print("Experiment {} Epoch {} saved".format(self.experiment_name,
                                                    self.epoch_num))
        self.epoch_num += 1

    def epoch_metrics(self, stats):
        # min_score,q1_score,mean_score,q3_score,max_score
        scores = [ms.score for ms in stats]
        avg_score = sum(scores) / float(len(scores))
        quartiles = numpy.percentile(scores,
                                     [0, 25, 75, 100],
                                     overwrite_input=True)
        return (",".join([str(quartiles[0]), str(quartiles[1]), str(avg_score),
                          str(quartiles[2]), str(quartiles[3])])
                + ",")

    def epoch_stats(self, epoch_num=None, experiment_name=None):
        epoch_num = epoch_num if isinstance(epoch_num, int) else self.epoch_num
        experiment_name = experiment_name or self.experiment_name
        stats = None
        try:
            weights = self.load_generation(epoch_num=epoch_num,
                                           experiment_name=experiment_name)
            stats = self.run_generation(weights)
        except IOError:
            pass
        except Exception as e:
            print e
            print type(e)
        finally:
            return stats

    def experiment_stats(self, epoch_start=None, experiment_name=None,
                         redo_epochs=False):
        # Look for stats file in experiment folder
        # Load each epoch folder that isn't in the stats file if epoch_start is
        # None and start from after the last epoch in the file (assumes from 0)
        # Write to file for epochs not yet done, overwrite if redo_epochs=True
        # Write in CSV format - Check metrics functions and subclasses for data
        experiment_name = experiment_name or self.experiment_name
        epoch_n = epoch_start if isinstance(epoch_start, int) else 0
        new_file = True
        try:
            if redo_epochs:
                os.remove(self.stats_name(experiment_name))
            else:
                with open(self.stats_name(experiment_name), 'r') as f:
                    # File exists
                    new_file = False
                    if epoch_start is None:
                        # Count epochs already done
                        i = -1
                        for i, l in enumerate(f):
                            pass
                        # Epochs and lines count from zero, so having a header
                        # doesn't change the number of epochs based on number
                        # of lines
                        epoch_n = max(i, 0)
        except (OSError, IOError):
            pass
        with open(self.stats_name(experiment_name), 'a+') as sf:
            original_epoch = epoch_n
            if new_file:
                # If redo_epochs is True, new_file is also True
                sf.write(self.experiment_stats_header() + "\n")
            # Else we already tried to count the number of epochs done
            epoch_s = self.epoch_stats(epoch_num=epoch_n,
                                       experiment_name=experiment_name)
            while epoch_s:
                metrics = self.epoch_metrics(epoch_s)
                sf.write(str(epoch_n) + ",")
                sf.write(metrics + "\n")
                epoch_n += 1
                epoch_s = self.epoch_stats(epoch_num=epoch_n,
                                           experiment_name=experiment_name)
        print("Made stats for {} epochs".format(epoch_n - original_epoch))

    def experiment_stats_header(self):
        # This is the required order of stat output
        return "epoch_num,min_score,q1_score,mean_score,q3_score,max_score,"

    def get_model_score(self, score):
        """
        @score: An iterable of the parameters that the model will be scored on
        @return: A numerical value
        """
        return reduce(__add__, score)

    def get_save_file(self, idx, epoch_num=None, experiment_name=None):
        epoch_num = epoch_num if isinstance(epoch_num, int) else self.epoch_num
        experiment_name = experiment_name or self.experiment_name
        fname = self.save_name(idx,
                               experiment_name=experiment_name,
                               epoch_num=epoch_num)
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        return fname

    def load_generation(self, epoch_num=None, experiment_name=None):
        epoch_num = epoch_num if isinstance(epoch_num, int) else self.epoch_num
        experiment_name = experiment_name or self.experiment_name
        output = [None for _ in xrange(self.generation_size)]
        for idx in xrange(self.generation_size):
            output[idx] = self.load_single_net(
                idx, epoch_num=epoch_num, experiment_name=experiment_name)
        return output

    @classmethod
    def load_single_net(cls, idx, epoch_num, experiment_name):
        data_output = list()
        with numpy.load(cls.save_name(
                idx, epoch_num=epoch_num, experiment_name=experiment_name
                )) as data:
            key_indices = sorted([int(key.split('_')[1])
                                  for key in data.keys()])
            for k in key_indices:
                data_output.append(data['arr_{}'.format(k)])
        return data_output

    def make_next_generation(self, generation_weights, generation_stats):
        # Sort by descending score, highest score on top
        generation_stats = sorted(generation_stats,
                                  key=attrgetter('score'),
                                  reverse=True)
        # Take only top %
        cutoff_num = int(math.floor(self.generation_cutoff *
                                    self.generation_size))
        # We need at least two for sexual-style reproduction
        if cutoff_num < 2:
            cutoff_num = 2
        cutoff_stats = generation_stats[:cutoff_num]
        return self.breed_new_generation([generation_weights[m.weights_idx]
                                          for m in cutoff_stats],
                                         [m.score for m in cutoff_stats])

    def resume_experiment(self, experiment_name=None, epoch_num=None):
        self.epoch_num -= 1
        loaded_weights = self.load_generation(epoch_num=epoch_num,
                                              experiment_name=experiment_name)
        approx_stats = self.run_generation(loaded_weights)
        new_weights = self.make_next_generation(loaded_weights, approx_stats)
        self.epoch_num += 1
        return self.run_all_generations(new_weights)

    def run_all_generations(self, weights):
        # Run each generation by starting from original weights and breeding
        # successive generations, then return the stats of the last generation
        for generation_num in xrange(1, self.generations + 1):
            if self.debug:
                print "Running generation {}".format(generation_num)
            stats = self.run_generation(weights)
            if self.debug:
                scores = [ms.score for ms in stats]
                print("High score {}".format(max(scores)))
                print("Average score {}".format(
                    sum(scores) / float(len(scores))))
                if isinstance(self, ScoringTrainer):
                    invalids = [ms.score_parameters.invalid_moves
                                for ms in stats]
                    valids = [ms.score_parameters.valid_moves
                              for ms in stats]
                    print("Least invalid moves {}".format(min(invalids)))
                    print("Most invalid moves {}".format(max(invalids)))
                    print("Average invalid moves {}".format(
                        sum(invalids) / float(len(invalids))))
                    print("Least valid moves {}".format(min(valids)))
                    print("Most valid moves {}".format(max(valids)))
                    print("Average valid moves {}".format(
                        sum(valids) / float(len(valids))))
            if (generation_num % self.epoch_size == 0 and
                    (generation_num != 0 or self.epoch_size == 1)):
                self.do_epoch(weights)
            weights = self.make_next_generation(weights, stats)
        return stats

    def run_experiment(self):
        def make_net_weights():
            return Net(
                hidden_sizes=self.net_hidden_sizes,
                weights=None,
                inputs=self.net_inputs,
                outputs=self.net_outputs,
                weight_spread=self.net_spread,
                weight_middle=self.net_middle).weights
        if self.net_weights is None:
            weights = [make_net_weights() for _ in xrange(self.generation_size)]
        else:
            weights = [self.net_weights for _ in xrange(self.generation_size)]
        stats = self.run_all_generations(weights)
        return stats

    def run_generation(self, generation_weights):
        # Get stats on a single generation
        stats = [None for _ in xrange(self.generation_size)]
        for idx, weights in enumerate(generation_weights):
            stats[idx] = self.run_model(weights, idx)
        return stats

    def run_model(self, weights, weights_idx, num_iterations=None):
        """
        @return: A ModelStats tuple
        """
        num_iterations = num_iterations or self.iterations_per_model
        n = Net(hidden_sizes=self.net_hidden_sizes,
                weights=weights,
                inputs=self.net_inputs,
                outputs=self.net_outputs,
                weight_spread=self.net_spread,
                weight_middle=self.net_middle)
        score_params = self.test_net(n, num_iterations)
        return ModelStats(weights_idx=weights_idx,
                          score_parameters=score_params,
                          score=self.get_model_score(score_params))

    def save_generation(self, weights):
        # Weights is a list of lists of numpy.ndarrays
        for idx, weight_array in enumerate(weights):
            numpy.savez(self.get_save_file(idx), *weight_array)

    @classmethod
    def save_name(cls, idx, experiment_name, epoch_num):
        # The suffix is added by numpy.savez
        return os.path.join("experiments",
                            experiment_name,
                            str(epoch_num),
                            "{}.npz".format(str(idx)))

    def stats_name(self, experiment_name=None):
        experiment_name = experiment_name or self.experiment_name
        return os.path.join("experiments",
                            experiment_name,
                            "stats.csv")

    def test_net(self, net, num_iterations):
        """Must return an iterable of score parameters"""
        raise NotImplementedError()


class ScoringTrainer(GeneticNetTrainer):
    """
    Assumes that games have a score and both valid and invalid moves
    """

    DEFAULT_MODEL_GAMES = 5
    SCORE_OBJ = ModelScore

    def __init__(self,
                 # Training Parameters
                 games_per_model=DEFAULT_MODEL_GAMES,
                 **kwargs):
        super(ScoringTrainer, self).__init__(**kwargs)

        if (not isinstance(games_per_model, int) or
                games_per_model <= 0):
            raise ValueError('Number of games per model must be a positive'
                             ' integer')

        # Training Variables
        self.games_per_model = games_per_model

    def _calc_game_score(self, scores):
        return sum(scores) / float(len(scores))

    def _calc_invalid_moves(self, invalid_moves):
        return sum(invalid_moves) / float(len(invalid_moves))

    def _calc_valid_moves(self, valid_moves):
        return sum(valid_moves) / float(len(valid_moves))

    def epoch_metrics(self, stats):
        base_metrics = super(ScoringTrainer, self).epoch_metrics(stats)
        # min_invalid,q1_invalid,mean_invalid,q3_invalid,max_invalid
        # min_valid,q1_valid,mean_valid,q3_valid,max_valid
        invalids = [ms.score_parameters.invalid_moves for ms in stats]
        valids = [ms.score_parameters.valid_moves for ms in stats]
        avg_invalids = sum(invalids) / float(len(invalids))
        avg_valids = sum(valids) / float(len(valids))
        i_quartiles = numpy.percentile(invalids,
                                       [0, 25, 75, 100],
                                       overwrite_input=True)
        v_quartiles = numpy.percentile(valids,
                                       [0, 25, 75, 100],
                                       overwrite_input=True)
        invalid_metrics = (",".join([str(i_quartiles[0]), str(i_quartiles[1]),
                                     str(avg_invalids),
                                     str(i_quartiles[2]), str(i_quartiles[3])])
                           + ",")
        valid_metrics = (",".join([str(v_quartiles[0]), str(v_quartiles[1]),
                                   str(avg_valids),
                                   str(v_quartiles[2]), str(v_quartiles[3])])
                         + ",")
        return base_metrics + invalid_metrics + valid_metrics

    def experiment_stats_header(self):
        base_header = super(ScoringTrainer, self).experiment_stats_header()
        # This is the required order of stat output
        return (base_header +
                "min_invalid,q1_invalid,mean_invalid,q3_invalid,max_invalid,"
                "min_valid,q1_valid,mean_valid,q3_valid,max_valid,")

    def test_net(self, net, num_iterations):
        """
        @net: An instantiated Net that can play the attached game
        @num_iterations: Max number of games to play
        @return: Score namedtuple
        """
        results = [None for _ in xrange(self.games_per_model)]
        for idx in xrange(self.games_per_model):
            results[idx] = self.test_net_once(net, num_iterations)
        return self.sum_score_objs(results)

    def sum_score_objs(self, results):
        return self.SCORE_OBJ(
            game_over=','.join(s.game_over for s in results),
            game_score=self._calc_game_score([s.game_score for s in results]),
            invalid_moves=self._calc_invalid_moves(
                [s.invalid_moves for s in results]),
            valid_moves=self._calc_valid_moves(
                [s.valid_moves for s in results]))

    def test_net_once(self, net, num_iterations):
        """
        Subclasses must play a single round of a game and return a score
        @return: Score namedtuple
        """
        raise NotImplementedError()


class BoardTrainer(ScoringTrainer):

    BOARD_OBJ = None
    NORMALIZE_BOARD = False

    def __init__(self,
                 # Board Parameters
                 board_dim_length=None,
                 board_dimensions=None,
                 full_board_ends_game=None,
                 invalid_move_ends_game=None,
                 # Training Parameters
                 normalize_board=None,
                 **kwargs):
        super(BoardTrainer, self).__init__(**kwargs)

        def get_default(name, val):
            return (val if val is not None
                    else getattr(self.BOARD_OBJ, name))

        self.board_dim_length = get_default('DEFAULT_LENGTH',
                                            board_dim_length)
        self.board_dimensions = get_default('DEFAULT_DIMENSIONS',
                                            board_dimensions)
        self.full_board_ends_game = get_default('FULL_BOARD_ENDS_GAME',
                                                full_board_ends_game)
        self.invalid_move_ends_game = get_default('INVALID_MOVE_ENDS_GAME',
                                                  invalid_move_ends_game)
        self.normalize_board = (
            normalize_board
            if normalize_board is not None
            else self.NORMALIZE_BOARD
            )
        self._board = None

        # Test to ensure nothing breaks
        self.BOARD_OBJ(
            dimensions=self.board_dimensions,
            length=self.board_dim_length,
            full_board_ends_game=self.full_board_ends_game,
            invalid_move_ends_game=self.invalid_move_ends_game)

    def get_board(self):
        if self._board:
            self._board.reset()
        else:
            self._board = self.BOARD_OBJ(
                dimensions=self.board_dimensions,
                length=self.board_dim_length,
                full_board_ends_game=self.full_board_ends_game,
                invalid_move_ends_game=self.invalid_move_ends_game)
        return self._board

    def test_net_once(self, net, num_iterations):
        """
        @return: Score namedtuple
        """
        board = self.get_board()
        return self.do_game(board, net, num_iterations)

    def do_game(self, board, net, num_iterations):
        """
        Subclasses must play a single round of a game and return a score
        @return: Score namedtuple
        """
        raise NotImplementedError()


class Genetic2048Trainer(BoardTrainer):

    BOARD_OBJ = Board2048

    def get_model_score(self, score):
        """
        @score: A named tuple of the Score2048 type
        @return: A numerical value
        """
        return score.game_score - score.invalid_moves

    def do_game(self, board, net, num_iterations):
        score = 0
        try:
            i = 0
            while i < num_iterations:
                i += 1

                if self.normalize_board:
                    data = board.normalize_board()
                else:
                    data = board._tiles.flatten()
                output = net.run(data)
                board.loop(output.argmax(), suppress_invalid=True)
        except GameOverOverflow as goo:
            reason = 'Overflow'
            score = 1 + int(str(goo))
        except GameOver as go:
            reason = 'Game Over'
            score = int(str(go))
        if not score:
            # Ended with iterations, not game filling up
            reason = 'Time Up'
            score = board.score()
        return self.SCORE_OBJ(
            game_over=reason,
            game_score=score,
            invalid_moves=board.invalid_moves,
            valid_moves=board.valid_moves)


class CompetitiveTrainer(BoardTrainer):
    """
    This trainer matches two nets against each other to play their game
    The game must be a two-person game
    """

    # This player will be used to play one extra round of games for each
    # organism in the gene pool as a "baseline" to prevent convergence
    # It will be used as both first and second player
    BASE_PLAYER = None

    def combine_scores(self, weights_idx, scores_list):
        all_params = self.sum_score_objs(scores_list)
        return ModelStats(
            weights_idx=weights_idx,
            # This performs averaging over the two sets
            score=self.get_model_score(all_params),
            score_parameters=all_params,
            )

    def get_competition_partner(self, idx):
        # Seed format, but don't make pairs play each other twice
        if idx > self.generation_size / 2:
            partner = self.generation_size - idx
        else:
            partner = self.generation_size - idx - 1
        return partner if partner != idx else 0

    def run_generation(self, generation_weights):
        # Get stats on a single generation
        scores = [list() for _ in xrange(self.generation_size)]
        for idx, weights in enumerate(generation_weights):
            competitor = self.get_competition_partner(idx)
            score_out = self.run_model(
                weights,
                generation_weights[competitor]
                )
            # Each net should play two rounds of games, one as first player, one
            # as second player
            scores[idx].extend(score_out[0])
            scores[competitor].extend(score_out[1])
            # Standardize the net against a baseline player
            if self.BASE_PLAYER:
                naive_stats_one = self.run_model(
                    weights,
                    self.BASE_PLAYER,
                    naive_player=2)
                scores[idx].extend(naive_stats_one)
                naive_stats_two = self.run_model(
                    self.BASE_PLAYER,
                    weights,
                    naive_player=1)
                scores[idx].extend(naive_stats_two)
        return [self.combine_scores(idx, score_list)
                for idx, score_list in enumerate(scores)]

    def run_model(self,
                  weights_test,
                  weights_adversary,
                  num_iterations=None,
                  naive_player=None):
        """
        @return: (ModelScore_test, ModelScore_adversary) where each is a list
                 of ModelScore tuples
                 Or a single list of ModelScore tuples for the non-naive player
                 if one is naive
        """
        num_iterations = num_iterations or self.iterations_per_model
        if naive_player == 1:
            n1 = weights_test()
        else:
            n1 = Net(
                hidden_sizes=self.net_hidden_sizes,
                weights=weights_test,
                inputs=self.net_inputs,
                outputs=self.net_outputs,
                weight_spread=self.net_spread,
                weight_middle=self.net_middle)
        if naive_player == 2:
            n2 = weights_adversary()
        else:
            n2 = Net(
                hidden_sizes=self.net_hidden_sizes,
                weights=weights_adversary,
                inputs=self.net_inputs,
                outputs=self.net_outputs,
                weight_spread=self.net_spread,
                weight_middle=self.net_middle)
        # We score both nets
        score_params = self.test_net(n1, n2, num_iterations)
        if naive_player is None:
            return score_params
        else:
            return score_params[naive_player - 1]

    def test_net(self, net_test, net_adversary, num_iterations):
        """
        Scores the first net, net_test, by pitting it against the second net,
        net_adversary
        @return: (list of net_test Score namedtuples,
                  list of net_adversary Score namedtuples)
        """
        results_test = [None for _ in xrange(self.games_per_model)]
        results_adversary = [None for _ in xrange(self.games_per_model)]
        for idx in xrange(self.games_per_model):
            results_test[idx], results_adversary[idx] = self.test_net_once(
                net_test, net_adversary, num_iterations)
        return (
            results_test,
            results_adversary,
            )

    def test_net_once(self, net_test, net_adversary, num_iterations):
        """
        @return: (net_test Score namedtuple, net_adversary Score namedtuple)
        """
        board = self.get_board()
        return self.do_game(board, net_test, net_adversary, num_iterations)


class TicTacToeTrainer(CompetitiveTrainer):

    BOARD_OBJ = BoardTTT
    BASE_PLAYER = NaiveTTTPlayer
    PLACE_TO_SCORE_MULTIPLIER = {
        'win': 1,
        'loss': 0,
        'draw': 0.5,
        }

    def do_game(self, board, net_test, net_adversary, num_iterations):
        player_to_move = 1
        player_one_place = 'draw'
        player_two_place = 'draw'
        reason = 'No Game'
        try:
            i = 0
            while i < num_iterations:
                i += 1

                if self.normalize_board:
                    data = board.normalize_board(player_to_move)
                else:
                    data = board._tiles.flatten()
                if player_to_move == 1:
                    output = net_test.run(data)
                else:
                    output = net_adversary.run(data)
                if board.loop(
                        output.argmax(),
                        player_to_move,
                        suppress_invalid=True):
                    # 2 player game
                    player_to_move = (player_to_move % 2) + 1
        except GameOverInvalid:
            reason = 'Invalid Move by Player {}'.format(player_to_move)
            # Treat it as a win if the other player made the invalid move
            # Treat it as a loss if this net made the invalid move
            player_one_place = 'win' if player_to_move == 2 else 'loss'
            player_two_place = 'win' if player_to_move == 1 else 'loss'
        except GameOver as go:
            reason = 'Game Ended'
            winning_player = int(str(go))
            if winning_player == 1:
                player_one_place = 'win'
                player_two_place = 'loss'
            elif winning_player == 2:
                player_one_place = 'loss'
                player_two_place = 'win'
            else:
                # Draw assumed by default
                pass
        score1 = self.PLACE_TO_SCORE_MULTIPLIER.get(player_one_place)
        score2 = self.PLACE_TO_SCORE_MULTIPLIER.get(player_two_place)
        return (
            self.SCORE_OBJ(
                game_over=reason,
                game_score=score1,
                invalid_moves=board.p1_invalid_moves,
                valid_moves=board.p1_valid_moves,
                ),
            self.SCORE_OBJ(
                game_over=reason,
                game_score=score2,
                invalid_moves=board.p2_invalid_moves,
                valid_moves=board.p2_valid_moves,
                ),
            )
