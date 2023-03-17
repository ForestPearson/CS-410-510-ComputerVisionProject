import cv2
import sys
import numpy as np
import math


def computeError(point1, point2, h):
    p1h = np.transpose([point1[0], point1[1], 1])
    p2h = np.transpose([point2[0], point2[1], 1])
    point1Predicted = np.dot(h, p1h)
    point1Predicted = point1Predicted/point1Predicted[2]
    error = np.linalg.norm(p2h-point1Predicted)
    return error


def computeHomography(x1, y1, x2, y2, x3, y3, x4, y4, xp1, yp1, xp2, yp2, xp3, yp3, xp4, yp4):
    A = np.array([[-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
                  [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
                  [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
                  [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
                  [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
                  [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
                  [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
                  [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]
                  ])
    U, S, V = np.linalg.svd(A, np.float32)
    H = V[-1:].reshape(3, 3)/V[-1, -1]
    return H


def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None
    check = False
    # to be completed ...
    for x in range(max_num_trial):  # Select seed group of matches
        first = list_pairs_matched_keypoints[np.random.randint(
            0, len(list_pairs_matched_keypoints))]
        second = list_pairs_matched_keypoints[np.random.randint(
            0, len(list_pairs_matched_keypoints))]
        third = list_pairs_matched_keypoints[np.random.randint(
            0, len(list_pairs_matched_keypoints))]
        fourth = list_pairs_matched_keypoints[np.random.randint(
            0, len(list_pairs_matched_keypoints))]
        #print("\n", first[0][0], "2", first[1][0])
        h = computeHomography(first[0][0], first[0][1], second[0][0], second[0][1], third[0][0], third[0][1], fourth[0][0], fourth[0][1],
                              first[1][0], first[1][1], second[1][0], second[1][1], third[1][0], third[1][1], fourth[1][0], fourth[1][1])
        # h = computeHomography(first[1][0], first[1][1], second[1][0], second[1][1], third[1][0], third[1][1], fourth[1][0], fourth[1][1],
        #                      first[0][0], first[0][1], second[0][0], second[0][1], third[0][0], third[0][1], fourth[0][0], fourth[0][1])

        numOfInliners = 0
        for i in range(len(list_pairs_matched_keypoints)):
            error = computeError(
                list_pairs_matched_keypoints[i][0], list_pairs_matched_keypoints[i][1], h)
            if error < threshold_reprojtion_error:
                numOfInliners += 1

        if numOfInliners / len(list_pairs_matched_keypoints) > threshold_ratio_inliers:
            best_H = h
            check = True
            break
    if(check == False):
        print("\n Warning, no H met thresholds. No H selected.\n")
    return best_H


def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================

    sift = cv2.SIFT_create()
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray_1, None)
    kp2, des2 = sift.detectAndCompute(gray_2, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []

    for i in range(len(kp1)):
        candidates = []
        for j in range(len(kp2)):
            dist = np.linalg.norm(des1[i] - des2[j])
            candidates.append((dist, j))
        candidates.sort()
        if (candidates[0][0] / candidates[1][0]) < ratio_robustness:
            p = kp1[i]
            q = kp2[candidates[0][1]]
            p1x = p.pt[0]
            p1y = p.pt[1]
            p2x = q.pt[0]
            p2y = q.pt[1]
            list_pairs_matched_keypoints.append([[p1x, p1y], [p2x, p2y]])
    # print(len(list_pairs_matched_keypoints))
    return list_pairs_matched_keypoints


def ex_warp_blend_crop_image(img_1, H_1, img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    maskInput = np.ones(img_1.shape[0:2], np.float32)

    h = img_1.shape[0]
    w = img_1.shape[1]
    inverseH = np.linalg.inv(H_1)
    # print(H_1)
    canvasIm1 = np.zeros([3*h, 3*w, 3], dtype=np.float32)
    maskIm1 = np.zeros([3*h, 3*w], dtype=np.float32)

    for intY in range(-h, 2*h):
        for intX in range(-w, 2*w):
            dstCordinateH = np.array([intX, intY, 1.0], np.float32)
            srcCordinateH = np.dot(inverseH, dstCordinateH)
            srcCordinateStandard = srcCordinateH / srcCordinateH[2]
            #print("W", srcCordinateH[0], "w", w-1.0)
            #print("H", srcCordinateH[1], "h", h-1.0)
            #print("S", srcCordinateStandard)
            i = int(math.floor(srcCordinateStandard[0]))
            j = int(math.floor(srcCordinateStandard[1]))
            a = srcCordinateStandard[0] - i
            b = srcCordinateStandard[1] - j
            #print("\nj", j, "i", i)
            #print("H", h, "w", w)
            # if(srcCordinateH[0] >= w-1 or srcCordinateH[1] >= h-1 or srcCordinateH[0] <= 0 or srcCordinateH[1] <= 0):
            if(j >= h-1 or j <= 0 or i >= w-1 or i <= 0):
                continue
            canvasIm1[intY+h, intX+w] = (1-a)*(1-b) * img_1[j, i] + a*(
                1-b) * img_1[j, i+1] + (a*b*img_1[j+1, i+1]) + ((1-a)*b * img_1[j+1, i])

            maskIm1[intY+h, intX+w] = (1-a)*(1-b) * maskInput[j, i] + a*(
                1-b) * maskInput[j, i+1] + (a*b*maskInput[j+1, i+1]) + (1-a)*b * maskInput[j+1, i]

    canvasIm2 = np.zeros([3*h, 3*w, 3], dtype=np.float32)
    maskIm2 = np.zeros([3*h, 3*w], dtype=np.float32)

    canvasIm2[h:h*2, w:w*2] = img_2
    maskIm2[h:h*2, w:w*2] = 1.0

    img = canvasIm1 + canvasIm2
    mask = maskIm1 + maskIm2
    mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
    img = np.divide(img, mask)  # Gives zero warnings

    mask_check = 1.0-np.float32(mask[:, :, 0] > 0)
    check_h = np.sum(mask_check[:, :], 1)
    check_w = np.sum(mask_check[:, :], 0)
    left = np.min(np.where(check_w < h*3))
    right = np.max(np.where(check_w < h*3))

    bottom = np.min(np.where(check_h < w*3))
    top = np.max(np.where(check_h < w*3))

    img_panorama = img[bottom:top, left:right]
    #img_panorama = canvasIm1

    return img_panorama


def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(
        img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(
        list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1, H_1=H_1, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]

    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(
        img_panorama).clip(0.0, 255.0).astype(np.uint8))
