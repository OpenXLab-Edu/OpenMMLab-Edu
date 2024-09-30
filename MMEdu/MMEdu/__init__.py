from .Classification import MMClassification,pth_info
from .Detection import MMDetection
# from .Generation_Edu import MMGeneration
# from .Pose import MMPose
from .version import __version__

__all__ = [
    'MMClassification',
    'MMDetection',
    # 'MMPose',
    '__version__',
    # '__path__',
    # 'MMGeneration',
    'pth_info'
]
