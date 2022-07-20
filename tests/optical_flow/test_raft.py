import os

import pytest

from python_libs.optical_flow import RaftModel


class TestRaft:
    @pytest.fixture(scope="class", autouse=True)
    def model(self) -> RaftModel:
        model_path = os.path.join(os.getcwd(), "models", "raft-sintel.pth")
        model = RaftModel(model_path)
        return model

    def test_process_video(self, model: RaftModel):
        frame_time = 300
        width = 416
        height = 416
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")
        flow = model.process_video(video_path, frame_time, width, height)
        assert flow is not None
        # 1/8 of the original size (416)
        assert flow[9].shape == (52, 52, 2)

    def test_process_video_hdf5(self, model: RaftModel):
        frame_time = 300
        width = 416
        height = 416
        file_ext = "_flow_test.hdf5"
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")

        if os.path.isfile(f"{video_path}{file_ext}"):
            os.remove(f"{video_path}{file_ext}")

        path = model.process_video_hdf5(
            video_path, frame_time, width, height, f"{video_path}{file_ext}"
        )
        assert path is not None

    def test_process_dataset(self, model: RaftModel):
        frame_time = 300
        width = 416
        height = 416
        file_ext = "_flow_test.hdf5"
        list_file = os.path.join(os.getcwd(), "tests", "optical_flow", "list.txt")
        root_path = os.path.join(os.getcwd(), "data", "raw")
        with open(list_file, "r") as f:
            video_list = f.read().splitlines()
            for video_name in video_list:
                video_path = os.path.join(root_path, video_name)
                if os.path.isfile(f"{video_path}{file_ext}"):
                    os.remove(f"{video_path}{file_ext}")

        model.process_dataset(
            frame_time,
            width,
            height,
            list_file,
            root_path,
            ext=file_ext,
            save_file=True,
        )
        with open(list_file, "r") as f:
            video_list = f.read().splitlines()
            for video_name in video_list:
                video_path = os.path.join(root_path, video_name)
                assert os.path.isfile(f"{video_path}{file_ext}") is True
                os.remove(f"{video_path}{file_ext}")
