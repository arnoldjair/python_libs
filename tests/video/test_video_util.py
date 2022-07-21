import os

from pymediainfo import MediaInfo

from python_libs.video import get_video_info_file
from python_libs.video.video_util import get_frames


class TestVideoUtil:
    def test_rotate(self):
        assert True is True

    def test_get_video_info(self):
        path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4.info")
        video_info = get_video_info_file(path)

        assert video_info is not None
        assert len(video_info) == 4

    def test_get_frames(self):
        path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")
        print(MediaInfo.parse(path))
        frames = get_frames(path)
        assert frames is not None
