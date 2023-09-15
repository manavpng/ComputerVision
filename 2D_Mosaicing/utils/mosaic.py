'''
This file is responsible for the actual functionality
'''
from cv2 import cvtColor, rotate, COLOR_BGR2GRAY, BFMatcher, RANSAC, findHomography, warpPerspective, VideoCapture, ROTATE_90_CLOCKWISE, imwrite, imread
from cv2.xfeatures2d import SURF_create as SURF
from numpy import array, float32, zeros

# Internal imports
from config import IMAGE, GENERAL, ERROR_MESSAGE, VIDEO

def mosaicing(image1, image2):
    '''
    Description: This function will do the image mosaicing
    '''
    # Converting image in grayscale format
    image1 = cvtColor(image1, COLOR_BGR2GRAY)
    image2 = cvtColor(image2, COLOR_BGR2GRAY)

    surf_obj = SURF()
    # Finding the keypoints and descriptors with SURF
    kp1, des1 = surf_obj.detectAndCompute(image1, None)
    kp2, des2 = surf_obj.detectAndCompute(image2, None)

    bf = BFMatcher()
    matches = bf.knnMatch(des1, des2, k=IMAGE.K_VALUE)

    # Apply ratio test
    good = []
    for match in matches:
        if match[GENERAL.ZERO].distance < GENERAL.HALF * match[GENERAL.ONE].distance:
            good.append(match)
    maches = array(good)
    if len(matches[:, GENERAL.ZERO]) >= IMAGE.MATCH_THRESHOLD:
        src = float32([ kp1[m.queryIdx].pt for m in matches[:, GENERAL.ZERO] ]).reshape(-1,1,2)
        dst = float32([ kp2[m.trainIdx].pt for m in matches[:, GENERAL.ZERO] ]).reshape(-1,1,2)
        H, _ = findHomography(src, dst, RANSAC, 5.0)
    else:
        raise AssertionError(ERROR_MESSAGE.KEYPOINT_ERROR)

    dst = warpPerspective(image1, H, (image2.shape[GENERAL.ONE] + image1.shape[GENERAL.ONE], image2.shape[GENERAL.ZERO]))

    dst[GENERAL.ZERO:image2.shape[GENERAL.ZERO], GENERAL.ZERO:image2.shape[GENERAL.ONE]] = image2
    for k in range(dst.shape[GENERAL.ONE]):
        if (dst[:, k]).any() == zeros((dst.shape[GENERAL.ZERO], 1, 3)).any():
            dst = dst[:, GENERAL.ZERO:k, :]
            break
    return dst

def image_mosaic():
    '''
    Description: This function will do image mosaicing on input images
    '''
    image_1 = imread(IMAGE.IMAGE_1)
    image_2 = imread(IMAGE.IMAGE_2)
    final_image = mosaicing(image_1, image_2)
    imwrite(GENERAL.OUTPUT_IMAGE, final_image)
    return True

def video_mosaic():
    '''
    Description: This function will break the video in frames depending on the frame rate defined in config and will do 2D mosaicing of all the frames one by one
    '''
    capture_obj = VideoCapture(VIDEO.VIDEO_LOC)
    i = 1
    images = []
    while capture_obj.isOpened():
        ret, frame = capture_obj.read()
        if not ret:
            break
        if i%VIDEO.FRAME_RATE == 0:
            frame = rotate(frame, ROTATE_90_CLOCKWISE)
            images.append(frame)
            i+=1
    current_image = images[GENERAL.ZERO]
    for i in range(1, len(images)):
        current_image = mosaicing(images[i], current_image)

    imwrite(GENERAL.OUTPUT_IMAGE, current_image)
    return True
