import os

from dotenv import load_dotenv

from python_libs.dataset import Protocol


class TestProtocol:
    def test_get_test_protocol(self):
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        json_path = os.path.join(os.getcwd(), "configs", "unit_test.json")
        train, test = Protocol.get_protocol(json_path)
        assert train is not None
        assert test is not None
