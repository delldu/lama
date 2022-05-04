import image_patch

# image_patch.image_client("TAI", "images/*.png", "output")
# image_patch.image_server("TAI")

# image_patch.video_client("TAI", "scratch.mp4", "output/repaired.mp4")
# image_patch.video_server("TAI")

image_patch.image_predict("images/*.png", "output")
# image_patch.video_predict("scratch.mp4", "output/predict.mp4")
