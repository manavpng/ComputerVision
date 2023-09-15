'''
@author: manavpng
Start Date: 15/09/2023
'''

from argparse import ArgumentParser

# Internal Imports
from config import IMAGE, VIDEO, GENERAL
from utils.error_handeling import HANDEL_ERROR
from utils.mosaic import video_mosaic, image_mosaic

if __name__ == "__main__":
    # Create argument parser object
    parser = ArgumentParser()

    # Define the command line arguments and their types
    parser.add_argument('--input_v', default=VIDEO.VIDEO_LOC, type= str, help='Location of video to do mosaicing on')
    parser.add_argument('--input_i',  default=[IMAGE.IMAGE_1, IMAGE.IMAGE_2], type=str, nargs=2, help='Location of input images separated by space (<image1><SPACE><image2> )')
    parser.add_argument('--out_image', type= str, help='Location of output image')

    # Parse the command line arguments
    args = parser.parse_args()

    # Retrieve the values of the command line arguments
    image_1_location, image_2_location = args.input_i
    video_location = args.input_v
    output_image_path = args.out_image

    # Creating object for verification of input path
    error_obj = HANDEL_ERROR(image_1_location, image_2_location, video_location, output_image_path)
    if video_location != VIDEO.VIDEO_LOC:
        error_obj.verify_images()
        mode = 0
    if image_1_location != IMAGE.IMAGE_1 or image_2_location != IMAGE.IMAGE_2:
        error_obj.verify_video()
        mode = 1

    # Updating default variables
    IMAGE.IMAGE_1 = image_1_location
    IMAGE.IMAGE_2 = image_2_location
    VIDEO.VIDEO_LOC = video_location
    GENERAL.OUTPUT_IMAGE = output_image_path

    if mode == 0:
        image_mosaic()
    elif mode == 1:
        video_mosaic()
