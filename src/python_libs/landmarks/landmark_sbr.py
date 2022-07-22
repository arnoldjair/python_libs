from pathlib import Path

import numpy as np
import torch
from nptyping import NDArray

from .sbr.lib import load_configure
from .sbr.lib.datasets import GeneralDataset as Dataset
from .sbr.lib.models import obtain_model, remove_module_dict
from .sbr.lib.utils import get_model_infos
from .sbr.lib.xvision import transforms


class LandmarkSBR:
    def __init__(self, model_path, config_path, cpu=True):
        self.cpu = cpu
        if not cpu:
            assert torch.cuda.is_available(), "CUDA is not available."
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        self.snapshot = Path(model_path)
        assert self.snapshot.exists(), "The model path {:} does not exist"

        if cpu:
            self.snapshot = torch.load(self.snapshot, map_location="cpu")
        else:
            self.snapshot = torch.load(self.snapshot)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        param = self.snapshot["args"]
        eval_transform = transforms.Compose(
            [
                transforms.PreCrop(param.pre_crop_expand),
                transforms.TrainScale2WH((param.crop_width, param.crop_height)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        model_config = load_configure(config_path, None)
        self.dataset = Dataset(
            eval_transform,
            param.sigma,
            model_config.downsample,
            param.heatmap_type,
            param.data_indicator,
        )
        self.dataset.reset(param.num_pts)
        self.net = obtain_model(model_config, param.num_pts + 1)

        if not cpu:
            self.net = self.net.cuda()

        try:
            weights = remove_module_dict(self.snapshot["detector"])
        except Exception as e:
            print(e)
            weights = remove_module_dict(self.snapshot["state_dict"])

        self.net.load_state_dict(weights)

    def process_image(self, image, face) -> NDArray:
        if image is None:
            return None

        if face == [0, 0, 0, 0]:
            return np.zeros((3, 68))

        [image, _, _, _, _, _, cropped_size], meta = self.dataset.prepare_input(
            image, face
        )

        with torch.no_grad():
            if self.cpu:
                inputs = image.unsqueeze(0)
            else:
                inputs = image.unsqueeze(0).cuda()
            batch_heatmaps, batch_locs, batch_scos = self.net(inputs)
            flops, params = get_model_infos(self.net, inputs.shape)

        cpu = torch.device("cpu")

        np_batch_locs, np_batch_scos, cropped_size = (
            batch_locs.to(cpu).numpy(),
            batch_scos.to(cpu).numpy(),
            cropped_size.numpy(),
        )
        locations, scores = np_batch_locs[0, :-1, :], np.expand_dims(
            np_batch_scos[0, :-1], -1
        )

        scale_h, scale_w = cropped_size[0] * 1.0 / inputs.size(-2), cropped_size[
            1
        ] * 1.0 / inputs.size(-1)

        locations[:, 0], locations[:, 1] = (
            locations[:, 0] * scale_w + cropped_size[2],
            locations[:, 1] * scale_h + cropped_size[3],
        )
        prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)

        return prediction

    def process_batch(self, images, face_locs) -> NDArray:
        # Test if image is not None
        batch = []
        cropped_sizes = []
        predictions = []
        latest = False
        for image, face_loc in zip(images, face_locs):
            if not np.any(face_loc):
                latest = True
            else:
                [image, _, _, _, _, _, cropped_size], meta = self.dataset.prepare_input(
                    image, face_loc
                )
                batch.append(image.unsqueeze(0))
                cropped_sizes.append(cropped_size)

        batch = torch.cat(batch)

        with torch.no_grad():
            if self.cpu:
                inputs = batch
            else:
                inputs = batch.cuda()
            batch_heatmaps, batch_locs, batch_scos = self.net(inputs)
            flops, params = get_model_infos(self.net, inputs.shape)

        cpu = torch.device("cpu")
        np_batch_locs, np_batch_scos = (
            batch_locs.to(cpu).numpy(),
            batch_scos.to(cpu).numpy(),
        )
        for np_batch_loc, np_batch_sco, cropped_size in zip(
            np_batch_locs, np_batch_scos, cropped_sizes
        ):
            cropped_size = cropped_size.numpy()
            locations, scores = np_batch_loc[:-1, :], np.expand_dims(
                np_batch_sco[:-1], -1
            )
            scale_h, scale_w = cropped_size[0] * 1.0 / inputs.size(-2), cropped_size[
                1
            ] * 1.0 / inputs.size(-1)
            locations[:, 0], locations[:, 1] = (
                locations[:, 0] * scale_w + cropped_size[2],
                locations[:, 1] * scale_h + cropped_size[3],
            )
            prediction = np.concatenate((locations, scores), axis=1).transpose(1, 0)
            predictions.append(prediction)

        if latest is True:
            predictions.append(np.zeros((3, 68)))

        return predictions
