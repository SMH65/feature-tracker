import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd


def measure(directory):
    time = []
    disp = []
    data = []

    frm = 0
    FPS = 30
    param = 1
    sample_len = 30

    green = (0, 225, 0)
    red = (0, 0, 225)

    cap = cv2.VideoCapture(directory)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.01,
                           minDistance = 50,
                           blockSize = 14)


    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    id_dat = []

    while frm <= sample_len:
        ret, frame = cap.read()

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1
        good_old = p0

        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            id_dat.append([i, a])

        frm = frm + 1

    id_dat = np.array(id_dat)
    id_dat = id_dat.reshape(frm, len(good_new), 2)  # frame, id, elements
    arr = np.zeros((len(good_new), frm))
    for n in range(frm):
        for i in range(len(good_new)):
            arr[i][n] = id_dat[n][i][1]


    std_arr = []
    for i in arr:
        std_arr.append(np.var(i))

    k = 0
    feature_id = 0
    for i in std_arr:
        if i == min(std_arr):
            feature_id = k

        k = k+1

    while (1):
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1
        good_old = p0

        y_buffer = []

        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            if i == feature_id: # feature id
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), green, 2)
                frame = cv2.circle(frame, (int(a),int(b)), 5, red, -1)

                y_buffer.append(b)

        y_mean = np.mean(y_buffer)
        img = cv2.add(frame, mask)
        k = cv2.waitKey(1) & 0xff

        data.append(y_mean)
        disp = data[0] - data

        disp = disp * param
        time.append(frm / FPS)

        frm = frm + 1

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img)

        if k == 27:
            break
        # date the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()

    fig = plt.figure()
    fig.suptitle('kalman filter thershold', fontsize=20)
    plt.plot(time, disp, color='orange')
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (px)', fontsize=20)
    plt.legend()

    plt.savefig('Kalman_project/Data/result_org_harris.png')

    plt.show()


    return disp, time

Data = measure("Kalman_project/video/girder.mp4")

df = pd.DataFrame(Data[0])
df.to_csv("Kalman_project/Data/data_org_harris.csv", header=False, index=False)
