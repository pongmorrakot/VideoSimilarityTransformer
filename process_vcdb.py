import json
import os

import cv2

outpath = "vcdb/"

read_txt = False

if read_txt:
    folder_path = "/home/ubuntu/Desktop/vcd-transformer/vcdb_dataset/core_dataset/"

    folders = os.listdir(folder_path)

    arr = []

    for f in folders:
        curpath = folder_path + f + "/"
        files = os.listdir(curpath)
        for file in files:
            if file.endswith(".txt"):
                # read
                r = open(curpath + file, "r")
                lists = r.readlines()
                for l in lists:
                    line = l.rstrip().split(",")
                    line[0] = curpath + line[0]
                    line[1] = curpath + line[1]
                    arr.append(line)
    # print(arr)

    w = open("annotation.txt", "w+")
    w.write(json.dumps(arr))


def to_second(ts):
    time = ts.split(":")
    second = int(time[0])*60*60 + int(time[1])*60 + int(time[2])
    return second


def get_clip(vid_name, start, end, outdir, interval=30):
    print(vid_name + "\t" + outdir)
    vidcap = cv2.VideoCapture(vid_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    count = int(start * fps)
    end = int(end * fps)
    # interval = (end - start)*fps / 20
    # print(end - start)
    # print(interval)
    # interval = 2
    vidcap.set(1, count)
    success = True
    while success and count < end:
        if count % interval == 0:
            success, image = vidcap.read()
            try:
                cv2.imwrite(outdir + "frame%d.jpg" % count, image)  # save frame as JPEG file
            except:
                print("Ope")
            print("Frame Captured:\t" + str(count))
        else:
            success = vidcap.grab()
        count += 1


r = open("annotation.txt", "r")
r = r.read()
# print(r)
r = json.loads(r)

i = 0
for line in r:
    folder_name = "clip" + str(i) + "/"
    curpath = outpath + folder_name
    vid1_path = curpath + "vid1/"
    vid2_path = curpath + "vid2/"
    os.mkdir(curpath)
    os.mkdir(vid1_path)
    os.mkdir(vid2_path)
    get_clip(line[0], to_second(line[2]), to_second(line[3]), vid1_path)
    get_clip(line[1], to_second(line[4]), to_second(line[5]), vid2_path)
    i += 1






