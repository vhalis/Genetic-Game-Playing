from argparse import ArgumentParser
from random import choice as rand_choice

from numpy import array as np_array

from nboard import (
    GameOver,
    NBoard,
    )


class GameOverInvalid(GameOver):
    pass


class BoardTTT(NBoard):

    DEFAULT_DIMENSIONS = 2
    DEFAULT_LENGTH = 3
    FULL_BOARD_ENDS_GAME = True
    PLAYER_TO_TILE = {
        1: "X",
        2: "Y",
        }

    def __init__(self, player_to_tile=PLAYER_TO_TILE, *args, **kwargs):
        kwargs['dimensions'] = kwargs.get('dimensions', self.DEFAULT_DIMENSIONS)
        kwargs['full_board_ends_game'] = self.FULL_BOARD_ENDS_GAME
        kwargs['length'] = kwargs.get('length',  self.DEFAULT_LENGTH)
        super(BoardTTT, self).__init__(*args, **kwargs)
        self.player_to_tile = player_to_tile
        self.winning_player = 0
        self.p1_invalid_moves = 0
        self.p1_valid_moves = 0
        self.p2_invalid_moves = 0
        self.p2_valid_moves = 0

    def are_there_valid_moves(self):
        return not self.is_board_full()

    def check_for_win(self):
        # 8 lines exist that can be the same symbol
        # Check verticals and horizontals
        # This only works for two dimensions right now
        def check_winner(row):
            winner = reduce(lambda x, y: y if x == y else 0, row)
            if winner:
                self.winning_player = winner
        for row in self._tiles:
            check_winner(row)
        for col in self._tiles.T:
            check_winner(col)
        check_winner(self._tiles.diagonal())
        # Flip left/right
        check_winner(self._tiles[..., ::-1].diagonal())
        if self.winning_player:
            self.game_over()

    def display(self):
        tile_idx = 0
        for idx in self._indices:
            print " ",
            tile_val = self._tiles[idx]
            if tile_val == 0:
                print tile_idx,
            else:
                print self.player_to_tile[tile_val],
            tile_idx += 1
            if tile_idx % self.length == 0:
                print ""

    def move(self, position, player, suppress_invalid=False):
        if position not in self.empty_tiles():
            if not suppress_invalid:
                print "That space is taken"
            if player == 1:
                self.p1_invalid_moves += 1
            else:
                self.p2_invalid_moves += 1
            self.invalid_moves += 1
            return False
        self._tiles[self._indices[position]] = player
        if player == 1:
            self.p1_valid_moves += 1
        else:
            self.p2_valid_moves += 1
        self.valid_moves += 1
        return True

    def post_valid_move(self):
        if self.is_board_full():
            self.game_over(full_board=True)
        else:
            self.check_for_win()

    def score(self):
        return self.winning_player

    def game_over(self, invalid_move=False, **kwargs):
        if invalid_move:
            raise GameOverInvalid(self.score())
        else:
            super(BoardTTT, self).game_over(**kwargs)

    def normalize_board(self, player_to_move, reshape=False):
        tile_copy = self._tiles.flatten()
        tile_map = {
            0: 0,
            1: 1 if player_to_move == 1 else -1,
            2: 1 if player_to_move == 2 else -1,
            }
        for idx, tile in enumerate(tile_copy):
            tile_copy.put(idx, tile_map[tile])
        if reshape:
            return tile_copy.reshape(self._tiles.shape)
        else:
            return tile_copy


class NaiveTTTPlayer(object):
    """
    The goal of this class is to provide a baseline for the genetically selected
    neural nets to play against so that they don't converge on solutions that
    only work against themselves. For instance, prior experiments have bred
    organisms that make invalid moves to end the game and keep all scores low,
    and organisms that only know how to play two games of tic tac toe and so
    can only play against each other.
    """

    def run(self, board):
        available_moves = [idx for idx, sq in enumerate(board) if sq == 0]
        random_idx = rand_choice(available_moves)
        return np_array([1 if idx == random_idx else 0
                         for idx in xrange(len(board))])


class Game(object):

    def __init__(
            self,
            net_number=None,
            epoch_number=None,
            experiment_name=None,
            use_naive_opponent=False,
            play_solo=False,
            **kwargs):
        self.board = BoardTTT(**kwargs)
        self.opponent = None
        if use_naive_opponent:
            self.opponent = NaiveTTTPlayer()
        elif (not play_solo
                and epoch_number is not None
                and experiment_name is not None
                and net_number is not None):
            from nets import Net
            from trainer import GeneticNetTrainer
            try:
                net_data = GeneticNetTrainer.load_single_net(
                    idx=net_number,
                    epoch_num=epoch_number,
                    experiment_name=experiment_name)
                # Hardcoding is hard
                net_hidden_sizes = (3, 3)
                # 2D Tic Tac Toe sizes
                net_inputs = 9
                net_outputs = 9
                self.opponent = Net(
                    hidden_sizes=net_hidden_sizes,
                    weights=net_data,
                    inputs=net_inputs,
                    outputs=net_outputs)
            except IOError:
                pass

    def run(self):
        last_move = None
        last_last_move = None
        player_to_move = 1
        try:
            while True:
                self.board.display()
                print("{} to move".format(
                    self.board.player_to_tile[player_to_move]))

                if player_to_move == 2 and self.opponent:
                    # AI always see themselves as player one
                    data = self.board.normalize_board(player_to_move)
                    output = self.opponent.run(data)
                    print("{} to take {}".format(
                        self.board.player_to_tile[player_to_move],
                        output.argmax()))
                    if self.board.loop(
                            output.argmax(),
                            player_to_move,
                            suppress_invalid=True):
                        # 2 player game
                        player_to_move = (player_to_move % 2) + 1
                    else:
                        print("The AI made an invalid move and probably can't"
                              " find a valid move. Sorry. :(")
                        self.board.game_over(invalid_move=True)
                    continue

                s = raw_input().strip().lower()
                is_number = False
                temp_move = -1
                try:
                    temp_move = int(s)
                    is_number = True
                except ValueError:
                    pass
                print("")
                if s.startswith('m'):
                    print("Moves: {}".format(self.board.valid_moves))
                    continue
                elif s.startswith('u'):
                    if self.board.undo():
                        last_move = last_last_move
                    continue
                elif s.startswith('q'):
                    self.board.game_over()
                elif (s.startswith('h')
                        or not is_number):
                    print("Commands:")
                    print("- [H]elp (this menu)")
                    print("- [M]oves (# of moves)")
                    print("- [U]ndo (revert last move)")
                    print("- [Q]uit (leave game)")
                    print("- Type a number to place your symbol there")
                    continue

                if self.board.loop(temp_move, player_to_move):
                    last_last_move = last_move
                    last_move = temp_move
                    player_to_move = (player_to_move % 2) + 1
        except GameOver as go:
            self.board.display()
            print("Game over! {} wins".format(
                self.board.player_to_tile.get(int(str(go)),
                                              "Nobody")))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net_number', type=int, default=0, nargs='?')
    parser.add_argument('epoch_number', type=int, default=0, nargs='?')
    parser.add_argument('experiment_name', type=str, default='ttt', nargs='?')
    parser.add_argument('--naive', action="store_true",
                        help="Play against an opponent that makes random moves")
    parser.add_argument('--solo', action="store_true",
                        help="Play by yourself - Not used if naive is on")
    args = parser.parse_args()
    Game(net_number=args.net_number,
         epoch_number=args.epoch_number,
         experiment_name=args.experiment_name,
         use_naive_opponent=args.naive,
         play_solo=args.solo).run()
