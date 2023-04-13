class Event:
    def __init__(self, pos_array, point_array, score=0.0, bdc=0.0):
        self.pos_array = pos_array
        self.point_array = point_array
        self.score = score
        self.bdc = bdc