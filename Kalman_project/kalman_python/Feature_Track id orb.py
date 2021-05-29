import numpy as np
import cv2
import pandas as pd


def measure(directory):
    import time

    start = time.time()

    t = []
    disp = []
    data = []
    buffer_id = []
    buffer_coord = []
    buffer_st = []
    

    frm = 0
    FPS = 30
    param = 1

    green = (0, 225, 0)
    red = (0, 0, 225)

    cap = cv2.VideoCapture(directory)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=10, edgeThreshold=300)
    kp, des = orb.detectAndCompute(old_gray, None)
    p0 = cv2.KeyPoint_convert(kp)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


    print(time.time()-start)


    while (1):
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        for i in st:
            buffer_st.append(i)

        # Select good points
        good_new = p1
        good_old = p0

        
        
        # draw the tracks`
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), green, 2)
            frame = cv2.circle(frame, (int(a),int(b)), 5, red, -1)
            feature_id = str(i)
            cv2.putText(frame, feature_id, (int(a),int(b)), 0, 5, (255, 0, 0), 1)

            buffer_id.append(i)
            buffer_coord.append(b)

        img = cv2.add(frame, mask)
        k = cv2.waitKey(1) & 0xff

        t.append(frm / FPS)

        frm = frm + 1

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', img)

        if k == 27:
            break
        # date the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
    cv2.destroyAllWindows()

    
    df = pd.DataFrame(buffer_id)
    df.to_csv("Kalman_project/Data/data_id.csv", header=False, index=False)

    df = pd.DataFrame(buffer_coord)
    df.to_csv("Kalman_project/Data/data_coord.csv", header=False, index=False)

    df = pd.DataFrame(t)
    df.to_csv("Kalman_project/Data/time.csv", header=False, index=False)

    df = pd.DataFrame(buffer_st)
    df.to_csv("Kalman_project/Data/st.csv", header=False, index=False)

    return buffer_coord, t

#Data = measure("Kalman_project/video/girder.mp4")
Data, Time =measure("Kalman_project/video/test_video1.mp4")



