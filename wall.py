import numpy as np
import cv2
import time



def sampling(img, x, y, block_size, gap,back_ground_up, back_ground_down):
    boundaryFlag = 0
    tmp_max = []
    tmp_min = []
    uperLeft = [np.mean(img[y - block_size:y - gap, x - block_size:x - gap, 0]),
                np.mean(img[y - block_size:y - gap, x - block_size:x - gap, 1]),
                np.mean(img[y - block_size:y - gap, x - block_size:x - gap, 2])]
    uperLeft_max = [np.max(img[y - block_size:y - gap, x - block_size:x - gap, 0]),
                    np.max(img[y - block_size:y - gap, x - block_size:x - gap, 1]),
                    np.max(img[y - block_size:y - gap, x - block_size:x - gap, 2])]
    uperLeft_min = [np.min(img[y - block_size:y - gap, x - block_size:x - gap, 0]),
                    np.min(img[y - block_size:y - gap, x - block_size:x - gap, 1]),
                    np.min(img[y - block_size:y - gap, x - block_size:x - gap, 2])]
    if np.max(uperLeft_max) > back_ground_up or np.min(uperLeft_min) < back_ground_down:
        boundaryFlag = 1
    else:
        tmp_max.append(uperLeft_max)
        tmp_min.append(uperLeft_min)
        cv2.rectangle(img, (x -block_size , y - block_size), (x - gap, y - gap), (0, 0, 255), 1)

    uperRight = [np.mean(img[y - block_size:y - gap, x + gap:x + block_size, 0]),
                 np.mean(img[y - block_size:y - gap, x + gap:x + block_size, 1]),
                 np.mean(img[y - block_size:y - gap, x + gap:x + block_size, 2])]
    uperRight_max = [np.max(img[y - block_size:y - gap, x + gap:x + block_size, 0]),
                     np.max(img[y - block_size:y - gap, x + gap:x + block_size, 1]),
                     np.max(img[y - block_size:y - gap, x + gap:x + block_size, 2])]
    uperRight_min=  [np.min(img[y - block_size:y - gap, x + gap:x + block_size, 0]),
                     np.min(img[y - block_size:y - gap, x + gap:x + block_size, 1]),
                     np.min(img[y - block_size:y - gap, x + gap:x + block_size, 2])]
    if np.max(uperRight_max) > back_ground_up or np.min(uperRight_min) < back_ground_down:
        boundaryFlag = 1
    else:
        tmp_max.append(uperRight_max)
        tmp_min.append(uperRight_min)
        cv2.rectangle(img, (x + gap, y - block_size), (x + block_size, y - gap), (0, 0, 255), 1)

    downleft = [np.mean(img[y + gap:y + block_size, x - block_size:x - gap, 0]),
                np.mean(img[y + gap:y + block_size, x - block_size:x - gap, 1]),
                np.mean(img[y + gap:y + block_size, x - block_size:x - gap, 2])]
    downleft_max = [np.max(img[y + gap:y + block_size, x - block_size:x - gap, 0]),
                    np.max(img[y + gap:y + block_size, x - block_size:x - gap, 1]),
                    np.max(img[y + gap:y + block_size, x - block_size:x - gap, 2])]
    downleft_min = [np.min(img[y + gap:y + block_size, x - block_size:x - gap, 0]),
                    np.min(img[y + gap:y + block_size, x - block_size:x - gap, 1]),
                    np.min(img[y + gap:y + block_size, x - block_size:x - gap, 2])]
    if np.max(downleft_max) > back_ground_up or np.min(downleft_min) < back_ground_down:
        boundaryFlag = 1
    else:
        tmp_max.append(downleft_max)
        tmp_min.append(downleft_min)
        cv2.rectangle(img, (x - block_size, y + gap), (x - gap, y + block_size), (0, 0, 255), 1)

    downRight = [np.mean(img[y + gap:y + block_size, x + gap:x + block_size, 0]),
                 np.mean(img[y + gap:y + block_size, x + gap:x + block_size, 1]),
                 np.mean(img[y + gap:y + block_size, x + gap:x + block_size, 2])]
    downRight_max = [np.max(img[y + gap:y + block_size, x + gap:x + block_size, 0]),
                     np.max(img[y + gap:y + block_size, x + gap:x + block_size, 1]),
                     np.max(img[y + gap:y + block_size, x + gap:x + block_size, 2])]
    downRight_min = [np.min(img[y + gap:y + block_size, x + gap:x + block_size, 0]),
                     np.min(img[y + gap:y + block_size, x + gap:x + block_size, 1]),
                     np.min(img[y + gap:y + block_size, x + gap:x + block_size, 2])]
    if np.max(downRight_max) > back_ground_up or np.min(downRight_min) < back_ground_down:
        boundaryFlag = 1
    else:
        tmp_max.append(downRight_max)
        tmp_min.append(downRight_min)
        cv2.rectangle(img, (x + gap, y + gap), (x + block_size, y + block_size), (0, 0, 255), 1)

    return boundaryFlag, np.array(tmp_max), np.array(tmp_min)



def threshHold_interval(src,min,max):
    if min > max:
        tmp = max
        max = min
        min = tmp

    if min == 0 :
        min = -1

    if max == 255 :
        max = 256

    _, srcH = cv2.threshold(src, max, 255, cv2.THRESH_BINARY_INV)
    _, srcL = cv2.threshold(src, min, 255, cv2.THRESH_BINARY)
    cut = cv2.bitwise_and(srcH,srcL)

    return cut

if __name__ == '__main__':
    start = time.time()
    img = cv2.imread("images/timg.jpg", 1)
    #img = cv2.imread("images/6.jpg", 1)
    #img = cv2.imread("images/8.png", 1)
    #img = cv2.imread("images/timg05.jpg", 1)
    (b, g, r) = cv2.split(img)

    img_np = np.array(img[5:15,5:15,:])
    img_np = img_np.reshape(-1,1)
    print(img_np)
    img_draw = img.copy()

    '''
    cv2.imshow("b_ori",b)
    cv2.imshow("g_ori",g)
    cv2.imshow("r_ori",r)
    '''
    edges = cv2.Canny(r, 250, 255, apertureSize=3)

    cv2.imshow('edge', edges)

    minLineLength = 300
    maxLineGap = 150
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)

    block_size = 6
    back_ground_up = 200    #img_np[int(16 / 20 * img_np.shape[0])-1]
    back_ground_down = 0    #img_np[int(04 / 20 * img_np.shape[0])-1]
    print(back_ground_up,back_ground_down)
    gap = 2
    Uper = []
    Lower = []

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 1)
            sample = 10
            for j in range(sample + 1):
                x = int((j + 1) * (max(x1, x2) - min(x1, x2)) / (sample + 1) + min(x1, x2))
                y = int((j + 1) * (max(y1, y2) - min(y1, y2)) / (sample + 1) + min(y1, y2))
                boundaryFlag, tmp_max, tmp_min = sampling(img_draw, x, y, block_size, gap,back_ground_up, back_ground_down)
                #print(tmp_min,boundaryFlag,)
                if boundaryFlag == 1 and tmp_max.shape[0]  :
                    for x in range(tmp_max.shape[0]):
                        Uper.append(tmp_max[x,:])
                        Lower.append(tmp_min[x,:])


    if not Uper:
        exit('无法找到上限')


    Max = []
    Min = []

    uper = np.array(Uper)
    uper[:,0].sort()
    uper[:,1].sort()
    uper[:,2].sort()

    print(uper)
    Max = uper[int(19 / 20 * uper.shape[0])-1,:]

    #Max = np.max(Uper, axis=0)
    print(Max)

    lower = np.array(Lower)
    lower[:,0].sort()
    lower[:,1].sort()
    lower[:,2].sort()
    #print(lower)
    Min = lower[int(1/ 20 * lower.shape[0]),:]
    #Min = np.min(Lower, axis=0)
    print(Min)

    b_cut = threshHold_interval(b, Min[0], Max[0])
    g_cut = threshHold_interval(g, Min[1], Max[1])
    r_cut = threshHold_interval(r, Min[2], Max[2])

    bg = cv2.bitwise_and(b_cut, g_cut)
    bgr = cv2.bitwise_and(bg, r_cut)

    bgr_medianBlur=cv2.medianBlur(bgr,5)


    kernel = np.ones((3, 3), np.uint8)
    closing_bgr = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, kernel)
    # ret_clo, clo_bi = cv2.threshold(closing, 100, 255, cv2.THRESH_BINARY)
    #bgr_medianBlur = cv2.medianBlur(closing_bgr, 3)
    kernel = np.ones((11, 11), np.uint8)
    opening_bgr = cv2.morphologyEx(bgr_medianBlur, cv2.MORPH_CLOSE, kernel)

    end = time.time()
    print (end - start)

    result = cv2.merge((b_cut, g_cut, r_cut))

    cv2.imshow('line', img_draw)
    cv2.imshow('result', result)
    cv2.imshow('bgr',bgr)
    cv2.imshow('opening_bgr', opening_bgr)
    cv2.waitKey(0)


