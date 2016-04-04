import shutil
import traceback

from collections import OrderedDict
from StringIO import StringIO
from unittest import TestCase

import numpy

from algs import (
    add_at_idx,
    add_at_l_idx,
    get_class_hierarchy_attrs,
    sort_quickest_ascending,
    )
from game import (
    Board,
    GameOver,
    )
from nets import (
    Net,
    )
from serializer import (
    Serializer,
    SerializerInterface,
    )
from trainer import (
    GeneticNetTrainer,
    Genetic2048Trainer,
    ModelStats,
    )


class TestRunner(TestCase):

    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[0;33m"
    COLOUR_OFF = "\033[0m"

    def print_colour(self, msg, colour):
        print("{colour}{msg}{colour_off}".format(
            msg=msg, colour=colour, colour_off=self.COLOUR_OFF))

    def print_green(self, msg):
        self.print_colour(msg, self.GREEN)

    def print_red(self, msg):
        self.print_colour(msg, self.RED)

    def print_yellow(self, msg):
        self.print_colour(msg, self.YELLOW)

    def runTest(self):
        print("")
        self.print_green("Running tests for {}".format(type(self).__name__))
        tests = [func_name for func_name in dir(self)
                 if func_name.startswith("test")]
        self.print_green("Found {} tests".format(len(tests)))
        flaky_tests = []
        num_fails = 0
        num_flaky = 0
        num_skipped = 0
        for test_name in tests:
            try:
                test_fn = getattr(self, test_name)
                if not TestRunner.get_is_skipped(test_fn):
                    test_fn()
                else:
                    num_skipped += 1
            except Exception as e:
                if TestRunner.get_is_flaky(test_fn):
                    num_flaky += 1
                    flaky_tests.append("{}: {}".format(test_fn.__name__, str(e)))
                else:
                    num_fails += 1
                    self.print_red("{} failed:".format(test_name))
                    print(traceback.format_exc())
        print("")
        if num_fails > 0:
            self.print_red("{} tests failed".format(num_fails))
        if num_flaky > 0:
            self.print_yellow("{} flaky tests failed:".format(num_flaky))
            for test_name in flaky_tests:
                self.print_yellow("> {}".format(test_name))
        if num_skipped > 0:
            self.print_yellow("{} tests skipped".format(num_skipped))

    @staticmethod
    def get_is_flaky(fn):
        return getattr(fn,
                       '_TestRunner_flaky',
                       False)

    @staticmethod
    def get_is_skipped(fn):
        return getattr(fn,
                       '_TestRunner_skip_test',
                       False)

    @staticmethod
    def flaky(fn):
        setattr(fn,
                '_TestRunner_flaky',
                True)
        return fn

    @staticmethod
    def skip(fn):
        setattr(fn,
                '_TestRunner_skip_test',
                True)
        return fn


class AlgsTest(TestRunner):

    def test_can_add_at_idx(self):
        t_1 = (0,1,2,3)
        t_2 = (0,5,2,3)
        self.assertEqual(t_2, add_at_idx(t_1, 1, 4))
        l_1 = [0,1,2,3]
        l_2 = [0,5,2,3]
        self.assertEqual(l_2, add_at_l_idx(l_1, 1, 4))

    def test_can_get_class_attrs_in_proper_order(self):
        class Dummy(object):
            def __init__(self, prop):
                self.test = prop
        class A(object):
            dummy = Dummy([1,2,3])
            test = [1,2,3]
            nyan = 'cat'
        class B(object):
            dummy = Dummy([4,5,6])
            test = [4,5,6]
            nyan = 'is'
        class C(B, A):
            dummy = Dummy([7,8,9])
            test = [7,8,9]
            nyan = 'This'

        test_output = get_class_hierarchy_attrs(C, 'test')
        dummy_output = get_class_hierarchy_attrs(C, 'test', sub_attr='dummy')
        self.assertEqual(test_output, [7,8,9,4,5,6,1,2,3])
        self.assertEqual(test_output, dummy_output)
        test_output_2, nyan_output = (
            get_class_hierarchy_attrs(C, ['test', 'nyan']))
        self.assertEqual(test_output, test_output_2)
        self.assertEqual(nyan_output, ['This', 'is', 'cat'])

    def test_can_get_class_attr_that_doesnt_exist(self):
        class A(object):
            pass
        class B(A):
            test = [1,2,3]

        test_output = get_class_hierarchy_attrs(B, 'test')
        self.assertEqual(test_output, [1,2,3])
        none_output = get_class_hierarchy_attrs(B, ['empty', 'moreempty'])
        self.assertEqual(none_output, [[],[]])

    def test_can_get_class_attr_that_is_none(self):
        class A(object):
            test = None
        class B(A):
            test = [1,2,3]

        test_output = get_class_hierarchy_attrs(B, 'test')
        self.assertEqual(test_output, [1,2,3, None])

    def test_can_sort_by_quickest_ascending_position(self):
        # This is sorted by 1st position quickest ascending
        q_1 = [(0,0,0), (1,0,0),
               (0,0,1), (1,0,1),
               (0,1,0), (1,1,0),
               (0,1,1), (1,1,1)]
        # This is sorted by 2nd position quickest ascending
        q_2 = [(0,0,0), (0,1,0),
               (0,0,1), (0,1,1),
               (1,0,0), (1,1,0),
               (1,0,1), (1,1,1)]
        # This is sorted by 3rd position quickest ascending
        q_3 = [(0,0,0), (0,0,1),
               (0,1,0), (0,1,1),
               (1,0,0), (1,0,1),
               (1,1,0), (1,1,1)]
        qs = [q_1, q_2, q_3]
        for q in qs:
            for i in xrange(len(qs)):
                self.assertEqual(sort_quickest_ascending(q, i), qs[i])


class SerializerTest(TestRunner):

    @TestRunner.skip
    def test_can_serialize_to_string(self):
        junk_str = 'Iamalongstring:Sorta'
        class Temp(object):
            _serializer = SerializerInterface(
                serializable_attrs=['a','b','c'],
            )
            def __init__(self, c):
                self.c = c

            def __eq__(self, other):
                for attr_name in ['a', 'b', 'c']:
                    if (getattr(other, attr_name, None) !=
                            getattr(self, attr_name, None)):
                        return False
                return True

        output = StringIO()
        s = Serializer(stream=output, stream_mode='str')
        t1 = Temp(c=junk_str)
        s.serialize(t1)
        t2 = s.deserialize(Temp)
        self.assertEqual(t1, t2)


class BoardTest(TestRunner):

    def test_can_make_boards(self):
        Board()
        for dim in range(1, 4):
            Board(dimensions=dim, length=dim, seed=False)
        self.assertRaises(ValueError,
                          Board, 0)
        self.assertRaises(ValueError,
                          Board, 0, 0)
        self.assertRaises(ValueError,
                          Board, 1, 0)
        self.assertRaises(ValueError,
                          Board, -1)
        self.assertRaises(ValueError,
                          Board, -1, -1)
        self.assertRaises(ValueError,
                          Board, 1, -1)

    def test_new_board_starts_with_two_squares(self):
        b = Board()
        self.assertFalse(b.is_board_full())
        self.assertEqual(len(b.empty_tiles()),
                         len(b._tiles.flatten())-2)

    def test_move_terminal_works(self):
        b = Board(seed=False)
        # Into (0,0) and (0,2)
        b._tiles.put(0, 1)
        b._tiles.put(2, 1)
        # This is not a terminal move because (0,1) is empty
        self.assertFalse(b._move_tile_terminal((0,2), (0,1)))
        # This is a terminal move because it results in a merge
        self.assertTrue(b._move_tile_terminal((0,1), (0,0)))
        self.assertEqual(b._tiles.take(0), 2)

    def test_move_row_works(self):
        b = Board(seed=False)
        # Make the top row 2 1 0 1
        b._tiles[0, :] = [2, 1, 0, 1]
        # Move to the left, so look at row left-to-right
        row_1 = ((0,0), (0,1), (0,2), (0,3))
        b._move_tile_row(row_1)
        self.assertEqual(b._tiles.take(0), 2)
        self.assertEqual(b._tiles.take(1), 2)
        b._move_tile_row(row_1)
        self.assertEqual(b._tiles.take(0), 4)

        # Make the second row 1 1 0 2
        b._tiles[1, :] = [1, 1, 0, 2]
        row_2 = ((1,0), (1,1), (1,2), (1,3))
        b._move_tile_row(row_2)
        # Expect 2 2 0 0
        self.assertEqual(b._tiles.take(4), 2)
        self.assertEqual(b._tiles.take(5), 2)
        b._move_tile_row(row_2)
        self.assertEqual(b._tiles.take(4), 4)

        # First column is now 2 2 0 0
        # Move up to merge
        col_1 = ((0,0), (1,0), (2,0), (3,0))
        b._move_tile_row(col_1)
        self.assertEqual(b._tiles.take(0), 8)

    def test_can_make_moves(self):
        b = Board(seed=False)
        # Make sure it's fully empty first
        self.assertEqual(len(b.empty_tiles()), 16)
        # Put a square in the top left corner
        val = 1
        pos = 0
        b._tiles.put(pos, val)
        # Move in the negative direction along the rows (up)
        # Then move in the negative direction along the columns (left)
        # Should do nothing, we are at the edge
        for d in (1, 3):
            b.move(d, suppress_invalid=True)
            self.assertEqual(b._tiles.take(pos), val)
        # Move in the positive direction along the rows (down)
        # This should move three rows
        b.move(0)
        pos += b.length*3
        self.assertEqual(b._tiles.take(pos), val)
        # At the bottom - movement should do nothing
        b.move(0, suppress_invalid=True)
        self.assertEqual(b._tiles.take(pos), val)
        # Move in the positive direction along the columns (right)
        # This should move three columns
        b.move(2)
        pos += 3
        self.assertEqual(b._tiles.take(pos), val)
        # At the right - movement should do nothing
        b.move(2, suppress_invalid=True)
        self.assertEqual(b._tiles.take(pos), val)

    def test_full_board_ends_game(self):
        b = Board(seed=False)
        for _ in xrange(16):
            b.make_tile()
        self.assertRaises(GameOver,
                          b.make_tile)

    def test_invalid_move_ends_game_when_set(self):
        b = Board(seed=False)
        b.loop(1, suppress_invalid=True)
        b.invalid_move_ends_game = True
        self.assertRaises(GameOver,
                          b.loop, 1, suppress_invalid=True)

    def test_can_merge_tiles(self):
        b = Board(seed=False)
        b._tiles[0, :] = [1, 1, 0, 0]
        # Tiles in first two columns - move left to merge
        b.move(3)
        self.assertEqual(b._tiles.take(0), 2)
        b._tiles.put(4, 2)
        # Tiles in first two rows - move up to merge
        b.move(1)
        self.assertEqual(b._tiles.take(0), 4)
        b._tiles.put(4, 2)
        # Different tiles in first two rows - move up to not merge
        b.move(1, suppress_invalid=True)
        self.assertEqual(b._tiles.take(0), 4)
        self.assertEqual(b._tiles.take(4), 2)

    def test_tiles_merge_in_proper_order(self):
        b = Board(seed=False)
        b._tiles[0, :] = [1, 1, 1, 0]
        b.move(3)
        for x, y in zip(b._tiles[0, :], [2, 1, 0, 0]):
            self.assertEqual(x, y)
        b._tiles[0, :] = [1, 1, 1, 1]
        b.move(3)
        for x, y in zip(b._tiles[0, :], [2, 2, 0, 0]):
            self.assertEqual(x, y)

    def test_can_normalize_board(self):
        b = Board(seed=False)
        for idx in xrange(b.num_elements):
            b._tiles.put(idx, 16)
        expected = numpy.ones((b.length, b.length), dtype=int)
        self.assertTrue(numpy.array_equal(b.normalize_board(reshape=True),
                                          expected))
        flat_expected = numpy.ones(b.num_elements, dtype=int)
        self.assertTrue(numpy.array_equal(b.normalize_board(), flat_expected))

        b = Board(seed=False)
        num = 2
        for idx in xrange(b.num_elements):
            b._tiles.put(idx, num)
            num *= 2
        expected = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        self.assertTrue(numpy.array_equal(b.normalize_board(reshape=True),
                                          expected))


class NetTest(TestRunner):

    def test_can_make_nets(self):
        n = Net(hidden_sizes=(8,), inputs=16, outputs=4)
        self.assertEqual(n.sizes, (16, 8, 4))
        self.assertEqual(len(n.weights), 2)
        self.assertEqual(n.weights[0].shape, (16, 8))
        self.assertEqual(n.weights[1].shape, (8, 4))

        self.assertRaises(ValueError,
                          Net, 3)
        self.assertRaises(ValueError,
                          Net, weights='haha')
        self.assertRaises(ValueError,
                          Net, weights=(1,))
        self.assertRaises(ValueError,
                          Net, inputs='lol')
        self.assertRaises(ValueError,
                          Net, outputs='rofl')

    def test_default_can_process_input(self):
        n = Net()
        data = numpy.random.random((n.inputs,))
        output = n.run(data)
        self.assertEqual(output.shape, (n.outputs,))

    def test_nets_can_use_callable_weights(self):
        n = Net(weights=Net.random_weights)
        data = numpy.random.random((n.inputs,))
        output = n.run(data)
        self.assertEqual(output.shape, (n.outputs,))

    def test_square_can_process_input(self):
        hidden_weights = (
            numpy.array([[3.0, 2.0, 0.5],
                         [3.0, 0.5, 2.0],
                         [0.5, 3.0, 2.0]]),
            numpy.array([[1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0]]),
        )
        n = Net(hidden_sizes=(3,), weights=hidden_weights, inputs=3, outputs=3)
        # Layer one:
        # 10.5 12 10.5
        # Layer two:
        # 33 33 33
        # Softmax:
        # 0.33 0.33 0.33
        data = numpy.array([1,2,3])
        output = n.run(data)
        for i in output:
            self.assertAlmostEqual(i, 0.333, 3)

    def test_default_can_play_2048(self):
        n = Net()
        b = Board()
        data = b._tiles.flatten()
        output = n.run(data)
        self.assertEqual(output.shape, (n.outputs,))
        b.move(output.argmax(), suppress_invalid=True)


class GeneticNetTrainerTest(TestRunner):

    def test_can_make_trainers(self):
        klass = GeneticNetTrainer
        klass()
        self.assertRaises(ValueError,
                          klass, generations=-1)
        self.assertRaises(ValueError,
                          klass, generations=0)
        self.assertRaises(ValueError,
                          klass, generation_cutoff=0.0)
        self.assertRaises(ValueError,
                          klass, generation_cutoff=1.0)
        self.assertRaises(ValueError,
                          klass, generation_cutoff=0)
        self.assertRaises(ValueError,
                          klass, generation_size=-1)
        self.assertRaises(ValueError,
                          klass, generation_size=0)
        self.assertRaises(ValueError,
                          klass, mutation_chance=1)
        self.assertRaises(ValueError,
                          klass, mutation_chance=1.01)
        self.assertRaises(ValueError,
                          klass, mutation_chance=0)
        self.assertRaises(ValueError,
                          klass, mutation_chance=-0.01)
        self.assertRaises(ValueError,
                          klass, mutation_difference=1)
        self.assertRaises(ValueError,
                          klass, mutation_difference=1.01)
        self.assertRaises(ValueError,
                          klass, mutation_difference=0)
        self.assertRaises(ValueError,
                          klass, mutation_difference=-0.01)
        self.assertRaises(ValueError,
                          klass, iterations_per_model=0)
        self.assertRaises(ValueError,
                          klass, iterations_per_model=1.0)
        self.assertRaises(ValueError,
                          klass, wubbalubbadubdub='So Rick')

    def test_trainer_respects_net_properties(self):
        iters = 77
        h_sizes = (4,)
        i_size = 17
        o_size = 44
        t = GeneticNetTrainer(
            iterations_per_model=iters,
            hidden_sizes=h_sizes,
            inputs=i_size,
            outputs=o_size,
            )
        sizes = (i_size,) + h_sizes + (o_size,)
        def test_net(net, num_iterations):
            self.assertEqual(num_iterations, iters)
            self.assertEqual(net.sizes, sizes)
            return (0,)
        t.test_net = test_net
        t.run_model(None)
        t2 = GeneticNetTrainer(
            iterations_per_model=iters,
            hidden_sizes=h_sizes,
            weights=Net.random_weights(sizes),
            inputs=i_size,
            outputs=o_size,
            )
        t2.test_net = test_net
        t2.run_model(None)

    def test_trainer_breeds_pair(self):
        t = GeneticNetTrainer(
            mutation_chance=0.0,
            )
        # Default 50% mixing chance, should never get 32 'heads' in a row
        num_genes = 32
        w_1 = numpy.array([1 for _ in xrange(num_genes)], dtype=int)
        w_2 = numpy.array([2 for _ in xrange(num_genes)], dtype=int)
        w_out = t.breed_pair(w_1, w_2)
        self.assertEqual(w_2.shape, w_out.shape)
        self.assertEqual(w_2.dtype, w_out.dtype)
        l_out = list(w_out.flatten())
        self.assertGreater(l_out.count(1), 0)
        self.assertGreater(l_out.count(2), 0)

    def test_trainer_mutates_pair_during_breeding(self):
        t = GeneticNetTrainer(
            mutation_chance=0.5,
            mutation_difference=1.0,
            )
        num_genes = 64
        w_1 = numpy.array([1.0 for _ in xrange(num_genes)])
        w_out = t.breed_pair(w_1, w_1)
        l_out = list(w_out.flatten())
        self.assertGreater(l_out.count(1.0), 0)
        self.assertNotEqual(l_out.count(1.0), num_genes)
        self.assertGreater(l_out.count(2.0), 0)
        self.assertGreater(l_out.count(0.0), 0)

    def test_trainer_breeds_generation(self):
        g_size = 3
        t = GeneticNetTrainer(
            generation_size=g_size,
            mutation_chance=0.0,
            )
        num_genes = 32
        w_1 = [numpy.array([1 for _ in xrange(num_genes)], dtype=int)]
        w_2 = [numpy.array([2 for _ in xrange(num_genes)], dtype=int)]
        ws_out = t.breed_new_generation([w_1, w_2])
        self.assertEqual(len(ws_out), g_size)
        for w in ws_out:
            # Organism has only one weight entry
            l = list(w[0].flatten())
            # No self-breeding - three new from population of two
            self.assertGreater(l.count(1), 0)
            self.assertGreater(l.count(2), 0)

    def test_trainer_makes_next_generation(self):
        g_size = 3
        # This trainer will only breed the top two
        t = GeneticNetTrainer(
            generation_size=g_size,
            generation_cutoff=0.5,
            mutation_chance=0.0,
            )
        num_genes = 32
        w_1 = ModelStats(
            weights=[numpy.array([1 for _ in xrange(num_genes)], dtype=int)],
            score=100,
            score_parameters=None,
            )
        w_2 = ModelStats(
            weights=[numpy.array([2 for _ in xrange(num_genes)], dtype=int)],
            score=10,
            score_parameters=None,
            )
        w_3 = ModelStats(
            weights=[numpy.array([3 for _ in xrange(num_genes)], dtype=int)],
            score=1,
            score_parameters=None,
            )
        ws_out = t.make_next_generation([w_1, w_2, w_3])
        self.assertEqual(len(ws_out), g_size)
        for w in ws_out:
            # Organism has only one weight entry
            l = list(w[0].flatten())
            # No self-breeding
            self.assertGreater(l.count(1), 0)
            self.assertGreater(l.count(2), 0)
            # Weakest member excluded from gene pool
            self.assertEqual(l.count(3), 0)


class Genetic2048TrainerTest(TestRunner):

    def test_can_run_single_model(self):
        t = Genetic2048Trainer(iterations_per_model=10)
        stats = t.run_model(None)
        self.assertGreater(stats.score_parameters.game_score, 0)

    def test_can_do_epoch(self):
        g_size = 2
        shutil.rmtree('experiments/TEST_EPOCH/', ignore_errors=True)
        t = Genetic2048Trainer(
            generations=1,
            generation_size=g_size,
            iterations_per_model=10,
            experiment_name='TEST_EPOCH',
            epoch_size=1,
            )
        t.run_experiment()
        loaded = t.load_generation(epoch_num=0)
        self.assertEqual(len(loaded), g_size)

    def test_can_run_single_generation(self):
        g_size = 10
        t = Genetic2048Trainer(
            generation_size=g_size,
            iterations_per_model=10,
            )
        weights = [None for _ in xrange(g_size)]
        stats = t.run_generation(weights)
        for s in stats:
            self.assertGreater(s.score_parameters.game_score, 0)

    def test_can_run_whole_experiment(self):
        g_size = 10
        t = Genetic2048Trainer(
            generations=3,
            generation_size=g_size,
            iterations_per_model=10,
            )
        s_out = t.run_experiment()
        self.assertEqual(len(s_out), g_size)

    def test_can_save_generation(self):
        g_size = 2
        t = Genetic2048Trainer(
            generations=1,
            generation_size=g_size,
            iterations_per_model=10,
            experiment_name='TEST_SAVE',
            )
        s_out = t.run_experiment()
        # s_out is a list of modelstats
        weights = [m.weights for m in s_out]
        t.save_generation(weights)
        loaded = t.load_generation()
        for a, b in zip(weights, loaded):
            for x, y in zip(a, b):
                numpy.testing.assert_array_almost_equal(x, y)

    @TestRunner.flaky
    def test_training_works(self):
        def test_if_invalid(stats):
            invalid = True
            for s in stats:
                if s.score_parameters.invalid_moves == 0:
                    invalid = False
            return invalid

        g_size = 5
        # TODO: Let the trainer take in previous output
        # TODO: Make Nets serializable
        t_1 = Genetic2048Trainer(
            generations=1,
            generation_size=g_size,
            iterations_per_model=10,
            )
        s_1_out = t_1.run_experiment()
        self.assertTrue(test_if_invalid(s_1_out))

        t_10 = Genetic2048Trainer(
            generations=10,
            generation_size=g_size,
            iterations_per_model=10,
            )
        s_2_out = t_10.run_experiment()
        # In ten generations, we should have at least one that doesn't make
        # any invalid moves when limited to 10 moves. But not guaranteed.
        self.assertFalse(test_if_invalid(s_2_out))


if __name__ == '__main__':
    AlgsTest().run()
    SerializerTest().run()
    BoardTest().run()
    NetTest().run()
    GeneticNetTrainerTest().run()
    Genetic2048TrainerTest().run()
