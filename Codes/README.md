# step1: segmentation&label
first frame————red channel————nucleus\
max/average————green channel————cell\
max(?)————blue channel————transcription sites\
\
**Note**: The current code still has some problems in threshold selection\
\
nucleus_seg_2.ipynb use multiostu to select threshold. The dim nuclei can be identified.
\
The number of cells obtained from the cell nucleus (red channel) can be used as prior knowledge for cell segmentation (green channel).
