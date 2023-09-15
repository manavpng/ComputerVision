'''
This file is to keep all the variables at a single location.
'''

class IMAGE:
    IMAGE_1 = ""
    IMAGE_2 = ""
    K_VALUE = 5
    MATCH_THRESHOLD = 4

class VIDEO:
    VIDEO_LOC = ""
    FRAME_RATE = 25

class ERROR_MESSAGE:
    KEYPOINT_ERROR = "Can't find enough keypoints"
    EXT_ERROR = "Invalid extension: Please use correct extension."
    FILE_NOT_AVAILABLE = "Invalid path: Input file path is not valid, please check again"

class GENERAL:
    ZERO = 0
    ONE = 1
    HALF = 0.5
    ALLOWED_IMAGE_EXT = [".png", ".jpg", ".jpeg"]
    ALLOWED_VIDEO_EXT = [".mp4", ".avi"]
    OUTPUT_IMAGE = "output.png"
