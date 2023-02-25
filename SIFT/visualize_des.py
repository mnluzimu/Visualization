import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans
import os


def get_sift(img, nfeatures=500, contrastThreshold=0.01, edgeThreshold=30, sigma=1.6):
    sift = cv.SIFT_create(nfeatures=nfeatures,
                          contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold,
                          sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


if __name__ == '__main__':
    root_path = r'../data/oxbuild_images-v1'
    img_fns = os.listdir(root_path)
    descriptors = []
    kps = []
    des_to_img = []
    for k, img_fn in enumerate(img_fns[:30]):
        img = cv.imread(os.path.join(root_path, img_fn))
        kp, des = get_sift(img)
        print(des.shape)
        descriptors.extend(des)
        kps.extend(kp)
        des_to_img.extend([k] * len(des))

    branchs = 750
    Model = MiniBatchKMeans(branchs,
                            init='k-means++',
                            n_init=10,
                            tol=0.0001,
                            verbose=1)
    Model.fit(descriptors)
    # print(Model)
    des_lists = [[] for i in range(branchs)]  # a 2 dim list built to save the ids in the child branches
    for i in range(len(descriptors)):
        des_lists[Model.labels_[i]].append(i)

    print(des_lists)
    lens = np.array([len(x) for x in des_lists])
    len_rank = np.argsort(-lens)
    des_lists = [des_lists[x] for x in len_rank]

    kp_imgs = []
    for i in range(10):
        des_list = des_lists[i][:20]
        row_kp_imgs = []
        for des in des_list:
            img_fn = img_fns[des_to_img[des]]
            kp = kps[des]
            pt = kp.pt
            pt = (int(pt[0]), int(pt[1]))
            print(pt)
            img = cv.imread(os.path.join(root_path, img_fn))
            print(img.shape)
            w = int(np.ceil(np.sqrt(kp.size)) * 3)
            kp_img = img[pt[1] - w: pt[1] + w, pt[0] - w: pt[0] + w, :]
            print(kp_img.shape)
            kp_img = cv.resize(kp_img, (50, 50))
            row_kp_imgs.append(kp_img)
        row_kp_imgs = np.concatenate(row_kp_imgs, axis=1)
        kp_imgs.append(row_kp_imgs)
    kp_imgs = np.concatenate(kp_imgs, axis=0)
    cv.imshow('kp_imgs', kp_imgs)

    cv.waitKey(0)

