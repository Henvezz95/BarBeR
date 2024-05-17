from abc import ABC, abstractmethod
import numpy.typing as npt

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, img:npt.NDArray) -> tuple[list[npt.NDArray], list[str], list[float | int | None]]:
        '''
        The detect method operates on a single image, in numpy array format
        The detect method has 3 outputs:
            - List of detected Bounding Boxes
            - List with the classes corresponding to each detection. Classes are strings ('1D', '2D', other)
            - List of confidence scores for each detection (if no confidence is available, it will be a None value)
        '''
        pass

    @abstractmethod
    def get_timing(self) -> int | float:
        pass
        