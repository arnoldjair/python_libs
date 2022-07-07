"""Record"""
import math
import os
from collections import OrderedDict
from typing import List

import h5py
import numpy as np
import pandas as pd
from nptyping import NDArray

from python_libs.video import get_video_info_file


class Record:
    """Record"""

    def __init__(self, video_path: str, label: int):
        self.video_path = video_path
        self.label = label
        self.landmarks = None
        self.flow = None

    def load(self, time: int = 0, every_frame: int = 1, samples=30) -> NDArray:
        """Load

        Args:
            time (int, optional): desired time between samples. Defaults to 0.
            every_frame (int, optional): number of frames between samples. Defaults to 1.

        Returns:
            NDArray: _description_
        """
        self.landmarks = self.get_landmarks(time=time, every_frame=every_frame)
        self.flow = self.get_flow()
        rep = self.landmarks
        # rep = np.concatenate((self.landmarks, np.array(
        #    self.flow).reshape((len(self.flow), -1), order='F')), axis=1)

        if np.shape(rep)[0] < samples:
            pad_length = samples - np.shape(rep)[0]
            rep = np.pad(rep, ((0, pad_length), (0, 0)), "constant", constant_values=1)

        return rep[:samples]

    def get_face_locs(self, index_col=0, face_path=None):
        """Gets face locs

        Args:
            index_col (int, optional): index col. Defaults to 0.
            face_path (str, optional): face locs path. Defaults to None.

        Returns:
            NDArray: face locations
        """

        curr_face_path = f"{self.video_path}.list"

        if face_path is not None:
            curr_face_path = face_path

        if os.path.isfile(curr_face_path):
            face_locs = pd.read_csv(
                curr_face_path,
                sep=" ",
                header=None,
                index_col=index_col,
                names=["x", "y", "w", "h"],
            )
            return face_locs

    def get_landmarks(self, time: int = 0, every_frame: int = 1) -> NDArray:
        """get landmarks

        Args:
            path (str): hdf5 file path
            time (int, optional): desired time between samples. Defaults to 0.
            every_frame (int, optional): number of frames between samples. Defaults to 1.

        Returns:
            Dict[int, NDArray]: landmarks
        """
        info = {}
        ret = OrderedDict()
        if os.path.isfile(f"{self.video_path}.hdf5"):
            with h5py.File(f"{self.video_path}.hdf5", "r") as hdf5_file:
                for current in hdf5_file.keys():
                    obj = hdf5_file[current]
                    info[int(current)] = np.array(obj)

        _, _, _, frame_rate = get_video_info_file(self.video_path)

        frame_space = 1

        if time == 0:
            if every_frame == 1:
                return info
            frame_space = every_frame
        else:
            frames_per_milisec = math.ceil(frame_rate) / 1000
            frame_space = math.ceil(time * frames_per_milisec)

        for i in range(0, len(info), frame_space):
            ret[i] = info[i]

        curr_landmarks = np.array(list(ret.values()))
        curr_landmarks = curr_landmarks[:, :2, :]
        curr_landmarks = curr_landmarks.reshape((len(curr_landmarks), -1), order="F")

        return curr_landmarks

    def get_flow(self) -> List[NDArray]:
        """Get flow individual

        Args:
            path (str): video path

        Returns:
            List[NDArray]: optical flow of shape (N, 52, 52, 3)
        """
        info = {}
        info[0] = np.zeros((52, 52, 2))
        if os.path.isfile(f"{self.video_path}_flow.hdf5"):
            with h5py.File(f"{self.video_path}_flow.hdf5", "r") as hdf5_file:
                for current in hdf5_file.keys():
                    obj = hdf5_file[current]
                    info[int(current)] = np.array(obj)

        ret = [info[a] for a in sorted(info)]
        ret.append(np.zeros((52, 52, 2)))
        return np.array(ret)
