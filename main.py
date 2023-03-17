import numpy as np
import torch
import kornia as K
import kornia.feature as KF
import cv2
import matplotlib.cm as cm
import pandas as pd
from matplotlib import pyplot as plt
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg


def siftTest(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img1 = cv2.drawKeypoints(img1, kp1, img1)
    cv2.imwrite('siftImageOne.jpg', img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    cv2.imwrite('siftImageTwo.jpg', img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    results = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            results.append([m])

    return des1, des2, results


def surfTest(img1, img2):
    surf = cv2.xfeatures2d.SURF_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    img1 = cv2.drawKeypoints(img1, kp1, img1)
    cv2.imwrite('surfImageOne.jpg', img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    cv2.imwrite('surfImageTwo.jpg', img2)

    results = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            results.append([m])

    return des1, des2, results


def briefTest(img1, img2):
    # sift = cv2.xfeatures2d.SIFT_create()
    # star = cv2.xfeatures2d.StarDetector_create()
    star = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    img1 = cv2.drawKeypoints(img1, kp1, img1)
    cv2.imwrite('briefImageOne.jpg', img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    cv2.imwrite('briefImageTwo.jpg', img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    results = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            results.append([m])

    return des1, des2, results


def loftrTest(img1, img2, type):
    matcher = LoFTR(config=default_cfg)
    if type == 'indoor':
        matcher.load_state_dict(torch.load(
            "weights/loftr_indoor.ckpt")['state_dict'])
    elif type == 'outdoor':
        matcher.load_state_dict(torch.load(
            "weights/loftr_outdoor.ckpt")['state_dict'])
    else:
        raise ValueError("Wrong image_type is given.")
    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

    batch = {'image0': img0, 'image1': img1}
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1,
                         color, mkpts0, mkpts1, text, path="LoFTR-result.pdf")

    print("batch items \n")
    # for k, v in batch.items():
    #    print (k)
    print("Keypoints for image 1\n")
    print(len(batch['mkpts0_f'].cpu().numpy().T[0]))
    return '\nend loftr'


def orbTest(img1, img2):
    orb = cv2.ORB_create()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    img1 = cv2.drawKeypoints(img1, kp1, img1)
    cv2.imwrite('orbImageOne.jpg', img1)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    cv2.imwrite('orbImageTwo.jpg', img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    results = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            results.append([m])
    return des1, des2, results


if __name__ == "__main__":
    # imageOne = 'TestOne.pgm'
    # imageTwo = 'TestTwo.pgm'
    img1 = cv2.imread('TestThree.jpg')
    img2 = cv2.imread('TestThree2.jpg')
    # imageOne = 'TestThree.jpg'
    # imageTwo = 'TestThree.jpg'
    imageOne = 'room-1.jpeg'
    imageTwo = 'room-2.jpeg'
    # print(torch.cuda.is_available())

    resultOne = orbTest(img1, img2)
    print("\n\nORB Results \nImage one keypoints: ", len(resultOne[0]), "\nImage Two keypoints: ", len(
        resultOne[1]), "\nMatches between: ", len(resultOne[2]))
    resultTwo = siftTest(img1, img2)
    print("\n\nSIFT Results \nImage one keypoints: ", len(resultTwo[0]), "\nImage Two keypoints: ", len(
        resultTwo[1]), "\nMatches between: ", len(resultTwo[2]))
    resultThree = briefTest(img1, img2)
    print("\n\nBRIEF Results \nImage one keypoints: ", len(resultThree[0]), "\nImage Two keypoints: ", len(
        resultThree[1]), "\nMatches between: ", len(resultThree[2]))
    resultFour = loftrTest(imageOne, imageTwo, 'indoor')

    #!!!!WARNING!!!!
    # REQUIRES PYTHON 3.7,3.6,3.5,3.4,2.7 AND opencv-contrib-python 3.2.2.17
    # Don't Enable otherwise
    #
    # resultFive = surfTest(img1, img2)
    # print("\n\nBRIEF Results \nImage one keypoints: ", len(resultFive[0]), "\nImage Two keypoints: ", len(
    #    resultFive[1]), "\nMatches between: ", len(resultFive[2]))
