import math
import os
import shutil

inpath = "vcdb_path/"
outpath = "vcdb/"

a = 0
for batch in os.listdir(inpath):
    print(batch)
    batchpath = inpath + batch + "/"

    folders = os.listdir(batchpath)
    num = math.floor(len(folders) / 2)
    for j in range(len(folders)):
        print("\t" + str(j) + "\t" + str(num) + "\t" + str(j + num) + "\t" + str(len(folders)))
        folderpath1 = batchpath + folders[j] + "/"
        folderpath2 = batchpath + folders[j + num] + "/"

        outpath1 = outpath + "clip" + str(a) + "/"
        a += 1
        outpath2 = outpath + "clip" + str(a) + "/"
        a += 1

        shutil.copytree(folderpath1 + "vid1/", outpath1 + "vid1/")
        shutil.copytree(folderpath2 + "vid2/", outpath1 + "vid2/")
        shutil.copytree(folderpath2 + "vid1/", outpath2 + "vid1/")
        shutil.copytree(folderpath1 + "vid2/", outpath2 + "vid2/")
        print("\t" + folderpath1 + "\t" + folderpath2)
        # i += 1
        if j >= num - 1:
            break
