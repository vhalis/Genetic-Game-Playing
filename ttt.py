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
        kwargs['dimensions'] = self.DEFAULT_DIMENSIONS
        kwargs['full_board_ends_game'] = self.FULL_BOARD_ENDS_GAME
        kwargs['length'] = kwargs.get('length') or self.DEFAULT_LENGTH
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
            raise GameOverInvalid(self.score)
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


class Game(object):

    def __init__(self, *args, **kwargs):
        self.board = BoardTTT(*args, **kwargs)

    def run(self):
        last_move = None
        last_last_move = None
        player_to_move = 1
        try:
            while True:
                self.board.display()
                print("{} to move".format(
                    self.board.player_to_tile[player_to_move]))
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
    Game().run()
