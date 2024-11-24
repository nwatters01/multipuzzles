"""Base puzzle class."""


class BasePuzzle:
    
    def __init__(self, pieces):
        self._pieces = pieces
        self._arrangements = []
        
    def add_arrangement(self, arrangement):
        self._arrangements.append(arrangement)