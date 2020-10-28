

class AlignmentUnmatchedAtomsError(Exception):
    def __init__(self, reference_array, other_array):

        message = f"Reference array has size {reference_array.size} while other array has size f{other_array.size}"

        # Call the base class constructor with the parameters it needs
        super().__init__(message)
