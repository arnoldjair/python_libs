import math
import os
from typing import Dict

import cv2
import imutils
import imutils.video.filevideostream as videostream
import numpy as np
from nptyping import NDArray
from pymediainfo import MediaInfo


def get_frames(path, time=0, every_frame=1, rotate_frame=False) -> Dict[int, NDArray]:
    assert os.path.isfile(path), "Invalid video file"

    video = videostream.FileVideoStream(path)
    assert video is not None, "Invalid video file"

    rotation, width, height, frame_rate = get_video_info(path)

    assert type(frame_rate) == float, "Invalid frame rate"

    print(f"Processing {path}: {rotation} - {width} - {height}")

    info = {}
    ret = {}
    video.start()
    curr_frame = 0

    while video.more():
        frame = video.read()
        if frame is not None:
            if rotate_frame:
                frame = rotate(frame, rotation)
            frame = imutils.opencv2matplotlib(frame)
        else:
            frame = np.ones((width, height, 3), dtype=int)
            if rotate_frame:
                frame = rotate(frame, rotation)

        info[curr_frame] = frame
        curr_frame = curr_frame + 1

    video.stop()

    frame_space = 1

    if time == 0:
        if every_frame == 1:
            return info
        else:
            frame_space = every_frame
    else:
        frames_per_milisec = math.ceil(frame_rate) / 1000
        frame_space = math.ceil(time * frames_per_milisec)

    for i in range(0, len(info), frame_space):
        ret[i] = info[i]

    return ret


def validate_video(path):
    media_info = MediaInfo.parse(path)
    valid = False
    for track in media_info.tracks:
        if track.track_type == "Video":
            valid = True
    if not valid:
        print("Invalid video file")
    return valid


def get_video_info(path):
    media_info = MediaInfo.parse(path)
    rotation = 0
    # TODO: generator expressions https://www.python.org/dev/peps/pep-0289/
    for track in media_info.tracks:
        if track.track_type == "Video":
            try:
                rotation = float(track.rotation)
                width = track.width
                height = track.height
                frame_rate = float(track.frame_rate)
                return rotation, width, height, frame_rate
            except Exception as ex:
                print(ex)
                return 0, 0, 0, 0


def get_video_info_file(path):
    with open(f"{path}.info", "r") as f:
        video_info = f.read().splitlines()[0].split(sep=" ")
        return [float(item) for item in video_info]


def rotate(image, rotation):
    if rotation is not None and rotation != 0 and image is not None:
        if int(rotation) == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif int(rotation) == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif int(rotation) == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image
