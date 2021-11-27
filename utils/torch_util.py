from copy import deepcopy
import json


def dir_convert_strings_to_unserializable_objects(dic, top_level=True):
    """
    Converts any strings with periods into objects
    :param dic:
    :param top_level:
    :return:
    """

    if top_level:
        dic = deepcopy(dic)

    for k, v in dic.items():
        if isinstance(v, dict):
            dir_convert_strings_to_unserializable_objects(v, top_level=False)
        elif isinstance(v, str):
            # Convert string to object that can be loaded
            dic[k] = str2obj(v)
    return dic


def str2obj(v):
    try:
        module, object_name = v.rsplit('.', maxsplit=1)
    except ValueError as e:
        bandu_logger.debug(f"Can't split {v}")
        return v
    try:
        bandu_logger.debug(f"Importing {module}")
        mod = importlib.import_module(module)
        return getattr(mod, object_name)
    except Exception as e:
        bandu_logger.debug(e)
        return v


def obj_to_string(obj):
    if isinstance(obj, list):
        try:
            return [obj_to_string(el) for el in obj]
        except AttributeError as e:
            return [el.__module__ + "." + el.__class__.__name__ for el in obj]
    else:
        bandu_logger.debug("obj")
        bandu_logger.debug(obj)
        return obj.__module__ + "." + obj.__name__


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False



def dir_convert_unserializable_objects_to_string(dic, top_level=True):
    if top_level:
        dic = deepcopy(dic)

    for k, v in dic.items():
        if isinstance(v, dict):
            dir_convert_unserializable_objects_to_string(v, top_level=False)
        elif not is_jsonable(v):
            # Convert function/object to a string that can be loaded
            try:
                # if isinstance(v, list):
                #     try:
                #         dic[k] = [el.__module__ + "." + el.__name__ for el in v]
                #     except AttributeError as e:
                #         dic[k] = [el.__module__ + "." + el.__class__.__name__ for el in v]
                # else:
                #     dic[k] = v.__module__ + "." + v.__name__
                dic[k] = obj_to_string(v)
            except AttributeError as e:
                bandu_logger.debug("Error")
                bandu_logger.debug(e)
                bandu_logger.debug(v)
                dic[k] = str(v)
    return dic


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = False