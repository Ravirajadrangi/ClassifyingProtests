PROJECT_ROOT="/home/gaspar/EyesOnProtests/"

DATA_ROOT="/DATA/SHARE/"

NEGATIVE_SAMPLES=DATA_ROOT+"raw/negative/negative_rescale/"
POSITIVE_SAMPLES=DATA_ROOT+"raw/positive/positive_rescale/"

DATASETS=DATA_ROOT+"datasets/"

ALL_DATA=DATASETS+"all_data/"
ALL_DATA_224=DATASETS+"all_data_224/"


### CHANGE this line to the correct directory
### for the vgg_features folder
VGG_FEATURES="datasets/vgg_features/"

RAW_POSTS=DATA_ROOT+"raw/posts/labelled_withimage.tsv"
POSTS=DATASETS+"posts/"


# Image specific used for initial image configuration. not used for VGG
dim = 100
depth = 3
pos_label=1
neg_label=0