import os
import shutil

folder_path = "vcdb/"
new_folder = "vcdb_batch/"
# os.mkdir(new_folder)
clips = os.listdir(folder_path)
arr = []
for i in range(60):
    arr.append([])
for i in range(len(clips)):
    index = i % 60
    arr[index].append(clips[i])
for i in range(60):
    path = new_folder + "batch" + str(i) + "/"
    os.mkdir(path)
    for j in arr[i]:
        orig_path = folder_path + j
        shutil.move(orig_path, path)
# print(arr)