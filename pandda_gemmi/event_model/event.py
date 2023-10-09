from ..interfaces import *


class EventBuild(EventBuildInterface):
    def __init__(self, build_path, score):
        self.build_path = build_path
        self.score = score


class Event(EventInterface):
    def __init__(self, pos_array, point_array, score=0.0, bdc=0.0, build=None):
        self.pos_array = pos_array
        self.point_array = point_array
        self.score = score
        self.bdc = bdc
        self.build = build