'''
This file is responsible for all the major and minor error handeling techniques for files.
'''

from os import _exit
from os.path import splitext, isfile, basename

from config import ERROR_MESSAGE, GENERAL


class HANDEL_ERROR():
    def __init__(self, first_image_location, second_image_location, video_location, output_image_location):
        self.first_image_location = first_image_location
        self.second_image_location = second_image_location
        self.video_location = video_location
        self.output_image_location = output_image_location

    @staticmethod
    def check_file_ext(file_path, file_type):
        '''
        Description: This function will check if the given file path exist in the system or not.
        '''
        if file_type == 0:
            if splitext(file_path)[GENERAL.ONE] not in GENERAL.ALLOWED_IMAGE_EXT:
                return False
            else:
                return True
        else:
            if splitext(file_path)[GENERAL.ONE] not in GENERAL.ALLOWED_VIDEO_EXT:
                return False
            else:
                return True

    def verify_images(self):
        '''
        Description: This function will verify the given inputs for image locations
        '''
        # Creating a list of all the images required to be checked
        image_paths = [self.first_image_location, self.second_image_location]
        for image_path in image_paths:
            if isfile(image_path):
                if not self.check_file_ext(image_path, GENERAL.ZERO):
                    print(ERROR_MESSAGE.EXT_ERROR)
                    print(f"Filename: {image_path}")
                    _exit(GENERAL.ZERO)
            else:
                print(ERROR_MESSAGE.FILE_NOT_AVAILABLE)
                print(f"Filename: {image_path}")
                _exit(GENERAL.ZERO)

    def verify_video(self):
        '''
        Description: This function will verify the given input for video location
        '''
        if isfile(self.video_location):
            if not self.check_file_ext(self.video_location, GENERAL.ONE):
                print(ERROR_MESSAGE.EXT_ERROR)
                print(f"Filename: {self.video_location}")
                _exit(0)
        else:
            print(ERROR_MESSAGE.FILE_NOT_AVAILABLE)
            print(f"Filename: {self.video_location}")
            _exit(0)
