"""Base puzzle class."""


class BasePuzzle:
    
    def __init__(self, pieces):
        self._pieces = pieces
        self._arrangements = []
        
    def add_arrangement(self, arrangement):
        self._arrangements.append(arrangement)
        
    def plot_arrangements(self):
        figures = []
        for arrangement in self._arrangements:
            figures.append(arrangement.plot())
        return figures