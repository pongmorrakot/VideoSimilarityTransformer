import cv2
import os


def capture(inpath, outpath, sequence_length=20, interval=30):
    vidcap = cv2.VideoCapture(inpath)

    success = True
    count = 0
    length = 0
    clip = 0
    folder = outpath + "clip" + str(clip) + "/"
    os.mkdir(folder)
    clip += 1

    while success:
        if length >= sequence_length:
            folder = outpath + "clip" + str(clip) + "/"
            os.mkdir(folder)
            clip += 1
            length = 0
        if count % interval == 0:
            success, image = vidcap.read()
            cv2.imwrite(folder + "frame%d.jpg" % count, image)     # save frame as JPEG file
            print("Frame Captured:\t" + str(count))
            length += 1
        else:
            success = vidcap.grab()
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
    return outpath

capture("/home/ubuntu/Desktop/vcd-transformer/New York City 4K - Diamond District - Midtown Manhattan - Driving Downtown - USA-QH7wzoIdHWs.mkv", "/home/ubuntu/Desktop/vcd-transformer/images/")