import cv2 as cv
import numpy as np
import os


def main():
    MAX_NUM = 100000
    root = r'F:\programming\Python\Visualization\data'
    image_path = os.path.join(root, 'oxbuild_images-v1')
    fnames = os.listdir(image_path)

    name1 = fnames[2]
    name2 = fnames[5]
    img1 = cv.imread(os.path.join(image_path, name1))
    img2 = cv.imread(os.path.join(image_path, name2))
    print(img1.shape)
    print(img1.dtype)

    # get SIFT descriptors and kps
    kp1, des1 = get_sift(img1, nfeatures=MAX_NUM, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)
    kp2, des2 = get_sift(img2, nfeatures=MAX_NUM, contrastThreshold=0.01, edgeThreshold=10, sigma=1.6)
    print('image name: {}, kp number: {}'.format(name1, len(kp1)))
    print('image name: {}, kp number: {}'.format(name2, len(kp2)))

    # draw kps
    draw_kp(img1, kp1)
    draw_kp(img2, kp2)

    cv.imshow(name1, img1)
    cv.imshow(name2, img2)

    # concatenate the images
    h = 1000
    img, rate1, rate2 = cat_img(img1, img2, h)

    # match kps with brute force
    tentatives = get_matches(des1, des2, 0.4)
    x1, y1, _ = img1.shape
    x2, y2, _ = img2.shape

    img = draw_matches(img, y1, kp1, kp2, rate1, rate2, tentatives)

    cv.imshow('img', img)

    k = cv.waitKey(0)
    # save the images
    if k == ord('s'):
        cv.imwrite(name1[:-4] + '_kps.png', img1)
        cv.imwrite(name2[:-4] + '_kps.png', img2)
        cv.imwrite(name1[:-4] + '_' + name2[:-4] + '_matches.png', img)

    print('>> end')


def get_sift(img, nfeatures=500, contrastThreshold=0.01, edgeThreshold=30, sigma=1.6):
    sift = cv.SIFT_create(nfeatures=nfeatures,
                          contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold,
                          sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def draw_kp(img, kp):
    for p in kp:
        cv.circle(img, (int(p.pt[0]), int(p.pt[1])), int(np.ceil(np.sqrt(p.size))), (0, 255, 255), 1)
        cv.circle(img, (int(p.pt[0]), int(p.pt[1])), 2, (0, 255, 255), -1)


def cat_img(img1, img2, h):
    x1, y1, _ = img1.shape
    x2, y2, _ = img2.shape

    img1 = cv.resize(img1, (int(h * y1 / x1), h))
    img2 = cv.resize(img2, (int(h * y2 / x2), h))
    rate1 = h / x1
    rate2 = h / x2

    img = np.concatenate([img1, img2], axis=1)
    return img, rate1, rate2


def get_matches(des1, des2, contrast):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    matchesMask = [False for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < contrast * n.distance:
            matchesMask[i] = True
    tentatives = [(m[0].queryIdx, m[0].trainIdx) for i, m in enumerate(matches) if matchesMask[i]]
    return tentatives


def draw_matches(img, y, kp1, kp2, rate1, rate2, tentatives):
    for m in tentatives:
        p1 = kp1[m[0]].pt
        p2 = kp2[m[1]].pt
        p1 = (int(p1[0] * rate1), int(p1[1] * rate1))
        p2 = (int(y + p2[0] * rate2), int(p2[1] * rate2))
        cv.line(img, p1, p2, (0, 255, 255), 1)
    return img


if __name__ == '__main__':
    main()