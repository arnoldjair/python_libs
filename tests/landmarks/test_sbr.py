import os

import pytest

from python_libs.image import load_image
from python_libs.landmarks import LandmarkSBR


class TestSBR:
    @pytest.fixture(scope="class", autouse=True)
    def model(self) -> LandmarkSBR:
        model_path = os.path.join(os.getcwd(), "models", "cpm_vgg16-epoch-049-050.pth")
        config_path = os.path.join(os.getcwd(), "configs", "SBRDetector.config")
        model = LandmarkSBR(model_path=model_path, config_path=config_path, cpu=False)
        return model

    def test_load_model(self, model: LandmarkSBR):
        assert model is not None, "None model"

    def test_process_image(self, model):
        assert model is not None, "None model"
        image_path = os.path.join(os.getcwd(), "data", "raw", "frontal.jpg")
        image = load_image(image_path)
        face_loc = [545, 682, 869, 1001]
        result = model.process_image(image, face_loc)
        assert result is not None
