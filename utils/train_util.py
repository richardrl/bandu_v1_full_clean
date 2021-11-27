from collections import OrderedDict

import torch
from utils import torch_util
from bandu.config import bandu_logger


def model_creator(config, train_dataset=None, device_id=0):
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise Exception
    bandu_logger.debug("use_cuda {}".format(use_cuda))


    config = torch_util.dir_convert_strings_to_unserializable_objects(config)

    models_container_dict = OrderedDict()

    i = 0
    while i < 100:
        if f"model{i}" in config.keys():
            model_class_as_str = torch_util.obj_to_string(config[f"model{i}"]['model_class'])
            model_kwargs = config[f"model{i}"]['model_kwargs']

            model = config[f"model{i}"]['model_class'](**model_kwargs)

            device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
            models_container_dict[config[f"model{i}"]["model_name"]] = model.to(device)
            i+=1
        else:
            break
    if i==0:
        bandu_logger.debug("No models found. Did you label the model names incorrectly?")
        raise NotImplementedError
    return models_container_dict


def init_weights(m):
    if hasattr(m, "weight"):
        torch.nn.init.uniform_(m.weight, -3e-9, 3e-9)
        m.bias.data.fill_(-3e-9)