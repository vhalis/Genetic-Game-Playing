import random

from collections import (
    OrderedDict,
    )

import numpy

from algs import (
    add_at_idx,
    sort_quickest_ascending,
    )

class GameOver(Exception):
    pass


class GameOverOverflow(GameOver):
    pass


class Board(object):
    """Tiles are stored in a numpy array"""
    DEFAULT_DIMENSIONS = 2
    DEFAULT_LENGTH = 4
    DIR_MAP = {
        'down': 0,
        'left': 3,
        'right': 2,
        'up': 1,
    }
    DOUBLE_CHANCE = 0.1
    FULL_BOARD_ENDS_GAME = True
    INVALID_MOVE_ENDS_GAME = False

    def __init__(self, dimensions=DEFAULT_DIMENSIONS, length=DEFAULT_LENGTH,
                 full_board_ends_game=FULL_BOARD_ENDS_GAME,
                 invalid_move_ends_game=INVALID_MOVE_ENDS_GAME,
                 seed=True):
        if dimensions <= 0 or length <= 0:
            raise ValueError('Dimensions and length must be positive integers')
        self.dims = dimensions
        self.directions = dimensions*2
        self.invalid_moves = 0
        self.length = length
        self.valid_moves = 0
        self.num_elements = self.length**self.dims
        self.full_board_ends_game = full_board_ends_game
        self.invalid_move_ends_game = invalid_move_ends_game
        # Initialize board to zeros
        self._tiles = None
        self._last_tiles = None
        self._init_tiles()
        if seed:
            self.make_tile()
            self.make_tile()

    def _init_tiles(self):
        dims = [self.length for _ in range(0, self.dims)]
        self._tiles = numpy.zeros(dims, dtype=int)
        self._last_tiles = None
        self._indices = list(numpy.ndindex(self._tiles.shape))

    def _make_val(self):
        return 1 if random.random() < (1.0 - self.DOUBLE_CHANCE) else 2

    def are_there_valid_moves(self):
        # Check all neighboring tiles for each tile
        for idx in self._indices:
            tile_val = self._tiles[idx]
            if tile_val == 0:
                return True
            for neighbor in (-1, 1):
                for dim in xrange(self.dims):
                    new_idx = add_at_idx(idx, dim, neighbor)
                    if new_idx[dim] < 0 or new_idx[dim] >= self.length:
                        continue
                    if (self._tiles[new_idx]
                            == tile_val):
                        return True
        return False

    def display(self):
        print(self._tiles)

    def empty_tiles(self):
        return [idx for idx, val in enumerate(self._tiles.flat)
                if val == 0]

    def game_over(self, overflow=False):
        if overflow:
            raise GameOverOverflow(self.score())
        else:
            raise GameOver(self.score())

    def is_board_full(self):
        return self._tiles.all()

    def loop(self, direction, suppress_invalid=False):
        if self.move(direction, suppress_invalid=suppress_invalid):
            self.make_tile()
            return True
        elif self.invalid_move_ends_game:
            self.game_over()
        elif not self.are_there_valid_moves():
            if self.full_board_ends_game:
                self.game_over()
        return False

    def make_tile(self):
        if self.is_board_full():
            self.game_over()
            return
        empty_tiles = self.empty_tiles()
        ridx = random.randint(0, len(empty_tiles)-1)
        self._tiles.put(empty_tiles[ridx], self._make_val())

    def move(self, direction, suppress_invalid=False):
        # Slide all tiles toward the specified direction
        if direction >= self.directions:
            raise ValueError('Invalid direction supplied to move: %s',
                             direction)
        temp_tiles = self._tiles.copy()
        # Two moves per direction
        dim = direction / 2
        # Even numbers move "down" an axis, like down along rows
        # Odd numbers move "up" an axis, like left along columns
        reverse = True if direction % 2 == 0 else False
        # Get slices along the desired dimension
        # Do this by ordering the indices of the dimension we want to move along
        # so they vary fastest, then taking one row at a time
        indices = sort_quickest_ascending(self._indices,
                                          dim,
                                          reverse=reverse)
        # Take only contiguous elements:
        bottom = 0
        while bottom < self.num_elements:
            self._move_tile_row(indices[bottom:bottom+self.length])
            bottom += self.length

        # Decide if the new board is a valid position
        # If something changed, then move was valid
        if not numpy.array_equal(self._tiles, temp_tiles):
            self._last_tiles = temp_tiles
            self.valid_moves += 1
            return True
        else:
            # Else, erase the move
            self._tiles = temp_tiles
            self.invalid_moves += 1
            if not suppress_invalid:
                print("Invalid move")
            return False

    def _move_tile_row(self, row_coords):
        # row_coords should be in reverse order of desired movement
        # e.g. to move left along a row, ((0,0), (0,1))
        # e.g. to move right along a row, ((0,1), (0,0))
        exclusive_bottom = -1
        for idx_from in xrange(1, self.length):
            cur_idx = idx_from
            for idx_to in xrange(idx_from-1, exclusive_bottom, -1):
                #import pdb; pdb.set_trace()
                if self._move_tile_terminal(row_coords[cur_idx],
                                            row_coords[idx_to]):
                    # Exclude the tile moved to from future moves on this row
                    exclusive_bottom = idx_to
                    break
                else:
                    # Follow a single tile all the way to an edge or collision
                    cur_idx = idx_to

    def _move_tile_terminal(self, coord_mover, coord_dest):
        # Returns true if movement should be terminal
        # This means that coord_dest should not be moved into again this move
        val_mover = self._tiles[coord_mover]
        val_dest = self._tiles[coord_dest]
        if val_mover == 0:
            # Nothing to move
            return False
        elif val_dest == 0:
            self._tiles[coord_dest] = val_mover
            self._tiles[coord_mover] = 0
            # Keep moving
            return False
        elif val_mover == val_dest:
            new_total = val_mover + val_dest
            if val_mover > new_total:
                # Overflow, game should end
                self.game_over(overflow=True)
            self._tiles[coord_dest] = new_total
            self._tiles[coord_mover] = 0
            # Stop moving on a merge
            return True
        else:
            # Collision, stop moving
            return True

    def normalize_board(self, reshape=False):
        tile_copy = self._tiles.flatten()
        unique_tiles = set(tile_copy)
        # Ensure nothing gets mapped to 0 except 0
        unique_tiles.add(0)
        ordered_tiles = sorted(unique_tiles)
        tile_map = {tile: idx for idx, tile in enumerate(ordered_tiles)}
        for idx, tile in enumerate(tile_copy):
            tile_copy.put(idx, tile_map[tile])
        if reshape:
            return tile_copy.reshape(self._tiles.shape)
        else:
            return tile_copy

    def undo(self):
        if self._last_tiles is not None:
            if numpy.array_equal(self._tiles, self._last_tiles):
                print("Can't undo more than one move at a time")
            else:
                self._tiles = self._last_tiles
                return True
        else:
            print("Can't undo first move")
        return False

    def score(self):
        return self._tiles.sum()


class Game(object):

    KEY_TO_DIR = OrderedDict((
        ('h', 'left'),
        ('j', 'down'),
        ('k', 'up'),
        ('l', 'right'),
    ))

    def __init__(self, *args, **kwargs):
        self.board = Board(*args, **kwargs)

    def run(self):
        last_move = None
        last_last_move = None
        try:
            while True:
                print("Last move: {}".format(last_move))
                self.board.display()
                s = raw_input().strip().lower()
                print("")
                if s.startswith('m'):
                    print("Moves: {}".format(self.board.valid_moves))
                    continue
                elif s.startswith('s'):
                    print("Game score: {}".format(self.board.score()))
                    continue
                elif s.startswith('u'):
                    if self.board.undo():
                        last_move = last_last_move
                    continue
                elif s.startswith('q'):
                    self.board.game_over()
                elif s.startswith('he') or s not in Game.KEY_TO_DIR:
                    print("Commands:")
                    print("- [He]lp (this menu)")
                    print("- [M]oves (# of moves)")
                    print("- [S]core (game score)")
                    print("- [U]ndo (revert last move)")
                    print("- [Q]uit (leave game)")
                    print("Movement:")
                    for k, v in Game.KEY_TO_DIR.iteritems():
                        print("- {} ({})".format(k, v))
                    print("")
                    continue

                temp_move = Game.KEY_TO_DIR[s]
                if self.board.loop(Board.DIR_MAP[temp_move]):
                    last_last_move = last_move
                    last_move = temp_move
        except GameOver as go:
            print("Game over! Moves: {} Score: {}".format(
                self.board.valid_moves, str(go)))


if __name__ == '__main__':
    Game().run()
