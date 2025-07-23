class Site:
    def __init__(self, event_ids, centroid, name=None, comment=None):
        self.event_ids = event_ids
        self.centroid = centroid
        self.name = name
        self.comment = comment