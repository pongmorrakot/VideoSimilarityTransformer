import os

path = "/home/ubuntu/Desktop/UCF_IMG/"
label_path = "UCF101-Class Index.txt"

def get_label(folder_name):
	return 0

f = open(label_path, "r")
w1 = open("train.txt", "w+")
w2 = open("test.txt", "w+")
labels = f.readlines()
table = {}
for label in labels:
	num, name = label.split()
	table[name] = num

folders = os.listdir(path)
for folder in folders:
	folder_path = path + folder + "/"
	videos = os.listdir(folder_path)
	i = 1
	for vid in videos:
		vid_path = folder_path + vid + "/"
		if i % 10 == 0:
			w2.write(str(table[folder]) + " " + vid_path + "\n")
		else:
			w1.write(str(table[folder]) + " " + vid_path + "\n")
		i += 1
		# print(str(table[folder]) + "\t" + vid_path)
