import dataclasses


@dataclasses.dataclass()
class SiteID:
    site_id: int

    def __hash__(self):
        return self.site_id

    def __int__(self):
        return self.site_id
