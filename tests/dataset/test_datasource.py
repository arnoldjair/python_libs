import os

from dotenv import load_dotenv

from python_libs.dataset import Datasource, Protocol


class TestDatasource:
    def test_get_datasource(self):
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        json_path = os.path.join(os.getcwd(), "configs", "unit_test.json")
        train, test = Protocol.get_records(json_path)
        dev_datasource = Datasource(train)
        test_datasource = Datasource(test)

        item_dev = dev_datasource.__getitem__(0)
        item_test = test_datasource.__getitem__(0)
        assert item_dev is not None
        assert item_test is not None
