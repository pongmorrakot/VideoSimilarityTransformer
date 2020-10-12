# VideoSimilarityTransformer
Note: Still need a lot of works

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
