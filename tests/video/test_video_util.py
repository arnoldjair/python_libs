from python_libs.video import get_video_info_file


class TestVideoUtil:
    def test_rotate(self):
        assert True is True

    def test_get_video_info(self):
        path = "/home/equipo/Insync/arnoldjair@gmail.com/GoogleDrive/Desarrollo/antispoofing_thesis/data/Rose/Dataset/13/Ps_NT_HW_g_E_13_132.mp4"
        video_info = get_video_info_file(path)

        assert video_info is not None
        assert len(video_info) == 4
