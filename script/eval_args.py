"""Shared task-args builder for eval_policy and sim_bridge env_wrapper."""
import os
import yaml


def build_task_args(task_name, task_config, ckpt_setting, *, root,
                    get_embodiment_config_fn, default_embodiment=None):
    """Read task config + resolve embodiment + camera dims, return args dict."""
    from envs import CONFIGS_PATH

    with open(os.path.join(root, "task_config",
                           f"{task_config}.yml"), "r",
              encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment") or default_embodiment
    if embodiment_type is None:
        raise RuntimeError("embodiment not specified in task config")

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"),
              "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"no embodiment file for {emb_type}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = _get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = _get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")
    args["left_embodiment_config"] = get_embodiment_config_fn(
        args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config_fn(
        args["right_robot_file"])

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"),
              "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    return args
