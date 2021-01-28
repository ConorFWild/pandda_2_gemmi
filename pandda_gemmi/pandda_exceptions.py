

class AlignmentUnmatchedAtomsError(Exception):
    def __init__(self, reference_array, other_array):

        message = f"Reference array has size {reference_array.size} while other array has size f{other_array.size}"

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

class ExceptionTooFewDatasets(Exception):
    def __init__(self, stage: str = "UNKNOWN"):

        message = f"Too few datasets! Failed at stage: {stage}"

        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        
        
class ExceptionUnmatchedAlignmentMarker(Exception):
    def __init__(self, residue_id) -> None:
        
        message = f"Found no reference atoms to compare against for residue: {residue_id}"
        
        super().__init__(message)
        
class ExceptionNoCommonAtoms(Exception):
    def __init__(self) -> None:
        
        message = f"Found no common atoms!"
        
        super().__init__(message)