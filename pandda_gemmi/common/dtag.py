import dataclasses


@dataclasses.dataclass()
class Dtag:
    dtag: str

    def __hash__(self):
        return hash(self.dtag)

    def __eq__(self, other):
        try:
            if self.dtag == other.dtag:
                return True
            else:
                return False
        except Exception as e:
            return False
