"""RAFT model
"""
import argparse
import logging
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from tqdm import tqdm

from python_libs.image import load_image, show_img
from python_libs.optical_flow.raft import RAFT, flow_viz
from python_libs.video import get_frames


class RaftModel:
    """RaftModel"""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        args = argparse.Namespace(model=model_path, small=False, mixed_precision=False)
        self.model = RAFT(args)
        self.pretrained_weights = torch.load(args.model)
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(self.pretrained_weights)
            self.model.to(self.device)
        else:
            self.device = "cpu"
            self.pretrained_weights = self.get_cpu_model(self.pretrained_weights)
            self.model.load_state_dict(self.pretrained_weights)
        logging.basicConfig(format="%(asctime)s %(message)s")

    def get_cpu_model(self, model):
        """Get cpu model

        Args:
            model (RAFT): model

        Returns:
            OrderedDict: The new model
        """
        new_model = OrderedDict()
        # get all layer's names from model
        for name in model:
            # create new name and update new model
            new_name = name[7:]
            new_model[new_name] = model[name]
        return new_model

    @staticmethod
    def visualize_flow(img, flo):
        """Show flow as image

        Args:
            img (NDArray): original image
            flo (NDArray): The up flow
        """
        # permute the channels and change device is necessary
        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)

        img_flo = np.concatenate([img, flo], axis=0)
        img_flo = load_image(img_flo)
        show_img(img_flo)

    @staticmethod
    def save_flow(flo, name, parent_path):
        """Save flow as image

        Args:
            flo (NDArray): The up flow
            name (str): The name
            parent_path (str): folder
        """
        img = load_image(flow_viz.flow_to_image(flo))
        final_path = os.path.join(parent_path, name)
        img.save(final_path)

    def process_video(
        self, path: str, frame_time: int, width: int, height: int, low=True
    ):
        """Process video

        Args:
            path (str): Video path
            frame_time (int): Time between frames
            width (int): Width
            height (int): Height
            low (bool, optional): Save the original flow. Defaults to True.

        Returns:
            Dict: Flow
        """
        frames = get_frames(path, time=frame_time)
        idx = list(frames.keys())
        counter = 0
        flow = {}
        with torch.no_grad():
            frame_1 = (
                torch.from_numpy(
                    np.array(load_image(frames.get(idx[0])).resize((width, height)))
                )
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            for i in range(1, len(idx) - 1):
                frame_2 = (
                    torch.from_numpy(
                        np.array(load_image(frames.get(idx[i])).resize((width, height)))
                    )
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                flow_low, flow_up = self.model(
                    frame_1, frame_2, iters=12, test_mode=True
                )
                if low:
                    flow[idx[i]] = flow_low[0].permute(1, 2, 0).cpu().numpy()
                else:
                    flow[idx[i]] = flow_up[0].permute(1, 2, 0).cpu().numpy()
                frame_1 = frame_2
                counter += 1
        return flow

    def process_video_hdf5(
        self,
        path: str,
        frame_time: int,
        width: int,
        height: int,
        hdf5_path: str,
        low=True,
    ) -> str:
        """Process video

        Args:
            path (str): _description_
            frame_time (int): _description_
            width (int): _description_
            height (int): _description_
            low (bool, optional): _description_. Defaults to True.

        Returns:
            str: _description_
        """
        if not os.path.isfile(hdf5_path):
            flow = self.process_video(path, frame_time, width, height, low)
            with h5py.File(hdf5_path, "w") as h_file:
                for index, curr_flow in tqdm(flow.items()):
                    h_file.create_dataset(str(index), data=curr_flow)
            return hdf5_path
        else:
            print(f"{hdf5_path} already processed")
            return None

    def process_dataset(
        self,
        frame_time: int,
        width: int,
        height: int,
        list_file: str,
        root_path: str,
        ext: str = "_flow.hdf5",
        save_file=False,
    ):
        """Process the dataset"""

        self.model.eval()
        logging.info("Opening list file: %s with root path: %s", list_file, root_path)
        with open(list_file, "r") as file:
            video_list = file.read().splitlines()
            for video_name in video_list:
                path = os.path.join(root_path, video_name)
                if os.path.isfile(f"{path}"):
                    if save_file:
                        self.process_video_hdf5(
                            path, frame_time, width, height, f"{path}{ext}"
                        )
                    else:
                        self.process_video(path, frame_time, width, height)
