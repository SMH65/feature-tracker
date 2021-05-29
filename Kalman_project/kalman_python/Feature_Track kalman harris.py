import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.core.fromnumeric import squeeze
import pandas as pd
from KalmanFilter import KalmanFilter


def measure(directory):

    # (dt, u, std_acc, std_meas, nfeature)
    kf = KalmanFilter(1/30, 1, 0.1, 0.001, 5)
    predictions = []
    predictions_mean = []
    nfeature = 5

    time = []
    disp = []
    data = []
    buffer = []
    init = []

    frm = 0
    FPS = 30

    green = (0, 225, 0)
    red = (0, 0, 225)

    cap = cv2.VideoCapture(directory)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = nfeature,
                           qualityLevel = 0.01,
                           minDistance = 300,
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
        y_buffer_mask = []

        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), green, 2)
            frame = cv2.circle(frame, (int(a),int(b)), 5, red, -1)

            y_buffer.append(b)
            buffer.append(b)

        if frm == 0:
            init = np.array(y_buffer)
        

        y_buffer = init - np.array(y_buffer)
        y_buffer_mask = y_buffer.tolist()

        #kalman process
        predictions.append(kf.predict())
        predictions_mean.append(np.mean(kf.predict()[0]))
       
        std_min = 10

        for i, k in enumerate(y_buffer):
            if abs((k - np.squeeze(np.mean(kf.predict()[0])))) > std_min:
                print(y_buffer_mask)
                print("at {0} frame, {1} triggered".format(frm, i))
                del y_buffer_mask[i]
                y_buffer[i] = np.mean(y_buffer_mask)
                init[i] = init[i] - k + np.mean(y_buffer_mask)

        z = kf.H * np.vstack((y_buffer,[0, 0, 0, 0, 0]))
        kf.update(z)
                


        y_mean = np.mean(y_buffer)
        img = cv2.add(frame, mask)
        k = cv2.waitKey(1) & 0xff

        data.append(y_mean)

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
    for i in range(nfeature):
        plt.plot(time, np.array(predictions).T[i][0], color='b', linewidth=1.5)
    plt.plot(time, np.squeeze(predictions_mean), label = 'mean of prediction', color='g', linewidth=1.5)
    plt.plot(time, data, label = 'measurement', color='r', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.legend()
    plt.show()

    df = pd.DataFrame(data)
    df.to_csv("Kalman_project/Data/data_kalman_harris.csv", header=False, index=False)


    return disp, time

Data = measure("Kalman_project/video/girder.mp4")


