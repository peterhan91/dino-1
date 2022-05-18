import cv2
import imageio
import numpy as np


DEL_PADDING_RATIO = 0.02  #used for del_black_or_white
CROP_PADDING_RATIO = 0.0  #used for my_crop_xyr

# del_black_or_white margin
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6


def detect_xyr(img_source):
    if isinstance(img_source, str):
        try:
            img = imageio.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
        if img is None:
            raise Exception("image file error:" + img_source)
    else:
        img = img_source


    width = img.shape[1]
    height = img.shape[0]

    myMinWidthHeight = min(width, height)  

    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    '''
    parameters of HoughCircles
    According to our test about fundus images, param2 = 30 is enough, too high will miss some circles
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=32,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            # x width, y height

            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True

    if not found_circle:
        # suppose the center of the image is the center of the circle.
        x = img.shape[1] // 2
        y = img.shape[0] // 2

        # get radius  according to the distribution of pixels of the middle line
        temp_x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)
    
    return (found_circle, x, y, r)

def my_crop_xyr(img_source, x, y, r, crop_size=None, mode='symmetric'):
    if isinstance(img_source, str):
        # img_source is a file name
        try:
            image1 = imageio.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    original_width = image1.shape[1]
    original_height = image1.shape[0]

    (image_height, image_width) = (image1.shape[0], image1.shape[1])

    img_padding = int(min(original_width, original_height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, image_width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, image_height - 1))

    if image_width >= image_height: 
        if image_height >= 2 * (r + img_padding):
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            image1 = image1[:, image_left:image_right]
    else:  
        if image_width >= 2 * (r + img_padding):
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            image1 = image1[image_bottom:image_top, :]

    if crop_size is not None:
        if image1.shape[0] > image1.shape[1]:
            # pad along axis 1 
            width = (image1.shape[0] - image1.shape[1]) // 2
            image1 = np.pad(image1, pad_width=((0, 0), (width, width), (0, 0)), mode=mode)
        else:
            width = (image1.shape[1] - image1.shape[0]) // 2
            image1 = np.pad(image1, pad_width=((width, width), (0, 0), (0, 0)), mode=mode)
        image1 = cv2.resize(image1, (crop_size, crop_size))

    return image1



def get_fundus(img_source, crop_size, mode='symmetric'):
    if isinstance(img_source, str):
        try:
            image1 = imageio.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:  #file not exists or orther errors
        raise Exception("image file error:" + img_source)

    min_width_height = min(image1.shape[0], image1.shape[1])

    if min_width_height < 100: # image too small
        return None

    (found_circle, x, y, r) = detect_xyr(image1)
    image1 = my_crop_xyr(image1, x, y, r, crop_size, mode)

    return image1