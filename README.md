# VideoSimilarityTransformer
Note: Still need a lot of works

Instruction:
 1. use extract_frame.py to extract frame; change the input path to directory of the dataset; the script should go over all subfolders and extract frames from each video. interval can be changed to extract more or less frame (e.g.interval=12 means grab a frame for every 12 frames of video).
 2. run transformer.py; change input path to the directory where extracted frames are located.

Progress:
- Action recognition seems to work fine with 68.5% top-one performance on UCF101 dataset

Description:
- transformer.py : the important one; currently being implemented as an action recognition transformer (tried on UCF101 and worked)
- train_prep.py : read directories of dataset and generate an annotation file that can be used for training and testing.
- extract_frame.py : extract frame from video
- process_vcdb.py : extract clips according to annotations from the VCDB dataset; has some redundant code

TODO:
- implement a proper train/test function
- test out video similarity task to see if it can yield similar level of performance
- see if the use of feature extractor can be eliminated
- etc.
