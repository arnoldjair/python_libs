import os

from python_libs.dataset.record import Record


class TestRecord:
    def test_load_record(self):
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")

        record = Record(video_path=video_path, label=0)
        assert record is not None

    def test_get_flow_individual(self):
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")
        record = Record(video_path=video_path, label=0)
        flow = record.get_flow()
        assert flow is not None

    def test_landmarks_individual(self):
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")
        record = Record(video_path=video_path, label=0)
        landmarks = record.get_landmarks(time=300)
        assert landmarks is not None

    def test_load(self):
        video_path = os.path.join(os.getcwd(), "data", "raw", "G_NT_5s_wg_E_10_1.mp4")
        record = Record(video_path=video_path, label=0)
        rep = record.load(time=300)
        assert rep is not None

        rep = record.load(time=300, samples=100)
        assert rep is not None
        assert rep.shape == (100, 5544)

        time = 300
        width = 416
        height = 416

        rep = record.load(time=time, samples=30, width=width, height=height)
