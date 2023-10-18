from ..interfaces import *


class EventBuild(EventBuildInterface):
    def __init__(self, build_path, ligand_key, score, centroid, bdc):
        self.build_path = build_path
        self.ligand_key = ligand_key
        self.score = score
        self.centroid = centroid
        self.bdc = bdc


class Event(EventInterface):
    def __init__(self, pos_array, point_array, centroid, score=0.0, bdc=0.0, build=None):
        self.pos_array = pos_array
        self.point_array = point_array
        self.centroid = centroid
        self.score = score
        self.bdc = bdc
        self.build = build