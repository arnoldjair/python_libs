import os

from dotenv import load_dotenv

from python_libs.dataset import Protocol, SingleDatasource


class TestDatasource:
    def test_get_test_datasource(self):
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        json_path = os.path.join(os.getcwd(), "configs", "unit_test.json")
        train, test = Protocol.get_protocol(json_path)
        dev_datasource = SingleDatasource(train)
        test_datasource = SingleDatasource(test)

        item_dev = dev_datasource.__getitem__(0)
        item_test = test_datasource.__getitem__(0)
        assert item_dev is not None
        assert item_test is not None
