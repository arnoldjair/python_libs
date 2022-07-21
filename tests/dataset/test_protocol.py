import os

from python_libs.dataset import Protocol


class TestProtocol:
    def test_get_test_protocol(self):
        json_path = os.path.join(os.getcwd(), "configs", "unit_test.json")
        train, test = Protocol.get_protocol(json_path)
        assert train is not None
        assert test is not None
