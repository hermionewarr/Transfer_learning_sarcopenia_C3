# Transfer_learning_sarcopenia_C3

This is the code for my master's project. 1 is loading and storing the slice of interest from a patients CT scan, segmented with ITK SNAP. 2 is training a pretrained machine learning model to segment skeletal muscle and find a value of SMA from those segments. The pretrained model used is FCN Resnet50. 3 is automatically segmenting patients C3 CT slice, using our retrained model. 
The file to create bone masks for each pateint is also included.
