# reid.py — ReID module placeholder.
#
# DeepSORT-based appearance embedding was removed in favour of a lighter HSV
# histogram comparison implemented directly in tracker.py:
#
#   _compute_histogram() — builds a 2-D Hue×Saturation histogram of a bbox crop.
#   _reid_check()        — compares new histograms against recently_counted using
#                          cv2.HISTCMP_CORREL; suppresses the count if the score
#                          exceeds REID_THRESHOLD.
#
# Both fast and accurate modes use this histogram approach.
# This file is retained so the package layout stays consistent and any future
# re-introduction of embedding-based ReID (e.g. MobileNet via DeepSORT) has a
# natural home without touching tracker.py imports.
