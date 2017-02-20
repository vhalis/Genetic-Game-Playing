import numpy


class GameOver(Exception):
    pass


class NBoard(object):
    """Tiles are stored in a numpy array"""
    DEFAULT_DIMENSIONS = 2
    DEFAULT_LENGTH = 4
    FULL_BOARD_ENDS_GAME = True
    INVALID_MOVE_ENDS_GAME = False

    BOARD_TILE_TYPE = int

    def __init__(self,
                 dimensions=DEFAULT_DIMENSIONS,
                 length=DEFAULT_LENGTH,
                 full_board_ends_game=FULL_BOARD_ENDS_GAME,
                 invalid_move_ends_game=INVALID_MOVE_ENDS_GAME):
        if dimensions <= 0 or length <= 0:
            raise ValueError('Dimensions and length must be positive integers')
        self.dims = dimensions
        self.invalid_moves = 0
        self.length = length
        self.valid_moves = 0
        self.num_elements = self.length**self.dims
        self.full_board_ends_game = full_board_ends_game
        self.invalid_move_ends_game = invalid_move_ends_game
        self._init_tiles()

    def _init_tiles(self):
        dims = [self.length for _ in xrange(0, self.dims)]
        self._tiles = numpy.zeros(dims, dtype=self.BOARD_TILE_TYPE)
        self._tile_proto = numpy.copy(self._tiles)
        self._last_tiles = None
        self._indices = list(numpy.ndindex(self._tiles.shape))

    def are_there_valid_moves(self):
        raise NotImplementedError()

    def display(self):
        print(self._tiles)

    def empty_tiles(self):
        return numpy.where(self._tiles.flat == 0)[0]

    def game_over(self, **kwargs):
        raise GameOver(self.score())

    def is_board_full(self):
        return self._tiles.all()

    def loop(self, *args, **kwargs):
        if self.move(*args, **kwargs):
            self.post_valid_move()
            return True
        elif self.invalid_move_ends_game:
            self.game_over(invalid_move=True)
        elif not self.are_there_valid_moves():
            if self.full_board_ends_game:
                self.game_over(full_board=True)
        return False

    def post_valid_move(self):
        pass

    def move(self):
        raise NotImplementedError()

    def reset(self):
        self._tiles = numpy.copy(self._tile_proto)
        self._last_tiles = None
        self.invalid_moves = 0
        self.valid_moves = 0

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
        raise NotImplementedError()
