import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")

# TORCHDYNAMO_DISABLE 延后到加载 pi0 模型前设置，避免干扰 curobo 的 torch.compile

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

import shutil

import imageio_ffmpeg as ffmpeg

ffmpeg_path = ffmpeg.get_ffmpeg_exe()

from generate_episode_instructions import *
from eval_args import build_task_args

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import pickle


EVAL_SECTION_DIVIDER = "─" * 62


def _print_eval_section(
    task_name: str,
    policy_name: str,
    *,
    eval_mode: str = "new",
    eval_result_name: str = "",
    history_path: str | None = None,
) -> None:
    label_width = 12
    mode_tag = "restore" if eval_mode == "resume" else "new"
    eval_text = f"{eval_result_name}({mode_tag})"
    print()
    print(EVAL_SECTION_DIVIDER)
    print(f"  {'Task Name'.ljust(label_width)} : \033[34m{task_name}\033[0m")
    print(f"  {'Policy Name'.ljust(label_width)} : \033[34m{policy_name}\033[0m")
    print(f"  {'Eval'.ljust(label_width)} : \033[92m{eval_text}\033[0m")
    if eval_mode == "resume" and history_path:
        print(f"  {'Restore Path'.ljust(label_width)} : \033[90m{history_path}\033[0m")
    print(EVAL_SECTION_DIVIDER)
    print()


def _connect_remote_if_needed(model, cfg: dict) -> None:
    if cfg.get("use_cpp") and cfg.get("stage") == 2 and hasattr(model, "connect_remote"):
        model.connect_remote()


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(
        parent_directory, "../task_config/_camera_config.yml"
    )

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _make_ffmpeg(video_size, out_path, scale_hd=1):
    """创建 ffmpeg subprocess，接收原始尺寸 rgb24 帧，可选 HD 放大输出。"""
    args = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        video_size,
        "-framerate",
        "10",
        "-i",
        "-",
    ]
    if scale_hd > 1:
        args.extend(["-vf", f"scale=iw*{scale_hd}:ih*{scale_hd}"])
    args.extend([
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        "23",
        out_path,
    ])
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    # diy yaml
    yaml_path = f"./task_config/config.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    USE_SPCAM_CAMERA = data["spcam"]
    RESTORE = data["restore"]
    HD = data["hd"]

    get_model = eval_function_decorator(policy_name, "get_model")

    args = build_task_args(task_name, task_config, ckpt_setting, root=".",
                           get_embodiment_config_fn=get_embodiment_config)
    embodiment_type = args.get("embodiment")

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(
        f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        if USE_SPCAM_CAMERA:
            camera_config = get_camera_config("Spcam_OBS")
            video_size = str(camera_config["w"] * 2) + "x" + str(camera_config["h"])
        else:
            camera_config = get_camera_config(args["camera"]["head_camera_type"])
            video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    if data.get("save_frame", False) and args.get("eval_video_save_dir") is None:
        save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = save_dir

    if args.get("eval_video_save_dir") is not None:
        data["eval_video_save_dir"] = str(args["eval_video_save_dir"])

    # output camera config
    print("============= Config =============\n")
    print(
        "\033[95mMessy Table:\033[0m "
        + str(args["domain_randomization"]["cluttered_table"])
    )
    print(
        "\033[95mRandom Background:\033[0m "
        + str(args["domain_randomization"]["random_background"])
    )
    if args["domain_randomization"]["random_background"]:
        print(
            " - Clean Background Rate: "
            + str(args["domain_randomization"]["clean_background_rate"])
        )
    print(
        "\033[95mRandom Light:\033[0m "
        + str(args["domain_randomization"]["random_light"])
    )
    if args["domain_randomization"]["random_light"]:
        print(
            " - Crazy Random Light Rate: "
            + str(args["domain_randomization"]["crazy_random_light_rate"])
        )
    print(
        "\033[95mRandom Table Height:\033[0m "
        + str(args["domain_randomization"]["random_table_height"])
    )
    print(
        "\033[95mRandom Head Camera Distance:\033[0m "
        + str(args["domain_randomization"]["random_head_camera_dis"])
    )

    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", "
        + str(args["camera"]["collect_head_camera"])
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", "
        + str(args["camera"]["collect_wrist_camera"])
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================\n")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(
        args["right_embodiment_config"]["arm_joints_name"][1]
    )

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = data.get("test_num", 100)
    topk = 1

    usr_args["cfg"] = data
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    model = get_model(usr_args)
    temp_path = f"{Path(save_dir).parent}/temp.pkl"
    st_seed, suc_num, suc_list = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        temp_path=temp_path,
        yaml_path=yaml_path,
        restore=RESTORE,
        hd=HD,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(args["eval_video_save_dir"], f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))
        file.write(f"\n{suc_list}")

    print(f"Data has been saved to {file_path}")

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # return task_reward


def eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    temp_path=None,
    yaml_path=None,
    restore=False,
    hd=1,
):
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    suc_list = []

    # Load initial config
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle restore logic
    eval_mode = "new"
    eval_id = None
    if os.path.exists(temp_path) and restore:
        # Restore previous eval session
        with open(temp_path, "rb") as f:
            # Remove old directory and restore saved path
            if os.path.exists(args["eval_video_save_dir"]):
                if os.path.isdir(args["eval_video_save_dir"]):
                    shutil.rmtree(args["eval_video_save_dir"])
                else:
                    os.remove(args["eval_video_save_dir"])
            
            args["eval_video_save_dir"] = pickle.load(f)
            Path(args["eval_video_save_dir"]).mkdir(parents=True, exist_ok=True)
            
            # Restore config from previous session if available
            test_yaml_path = os.path.join(args["eval_video_save_dir"], "config.yaml")
            if os.path.exists(test_yaml_path):
                with open(test_yaml_path, "r", encoding="utf-8") as config_f:
                    data = yaml.safe_load(config_f)
                print(f"\033[92mRestored config from last eval:\033[0m {test_yaml_path}")
            else:
                print(f"\033[33mCould not restore last config, using current config!\033[0m")
            
            # Restore eval status
            temp_restore_path = os.path.join(args["eval_video_save_dir"], "temp.pkl")
            if os.path.exists(temp_restore_path):
                # Read from restored directory's temp file
                with open(temp_restore_path, "rb") as status_f:
                    _ = pickle.load(status_f)
                    now_id = pickle.load(status_f)
                    suc_list = pickle.load(status_f)
            else:
                # Read from current temp_path file
                _ = pickle.load(f)
                now_id = pickle.load(f)
                suc_list = pickle.load(f)
            
            TASK_ENV.test_num = now_id
            succ_seed = now_id
            TASK_ENV.suc = len(suc_list)
            print(f"\033[92mRestored status from last eval:\033[0m {TASK_ENV.suc}/{TASK_ENV.test_num}")
            print(f"\033[92mCurrent success list: \033[0m{suc_list}")
            eval_mode = "resume"
            eval_id = now_id
    elif os.path.exists(temp_path):
        # Clear previous eval status
        os.remove(temp_path)

    eval_result_dir = args.get("eval_video_save_dir")
    eval_result_name = Path(eval_result_dir).name if eval_result_dir else "unknown"
    history_path = str(eval_result_dir) if eval_mode == "resume" and eval_result_dir else None

    _print_eval_section(
        args["task_name"],
        args["policy_name"],
        eval_mode=eval_mode,
        eval_result_name=eval_result_name,
        history_path=history_path,
    )
    print("Engine Starting...")

    # Ensure config.yaml exists in eval directory
    test_yaml_path = os.path.join(args["eval_video_save_dir"], "config.yaml")
    if not os.path.exists(test_yaml_path):
        Path(args["eval_video_save_dir"]).mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_path, test_yaml_path)

    # Define temp_restore_path for later use
    temp_restore_path = os.path.join(args["eval_video_save_dir"], "temp.pkl")

    args["cfg"] = data
    RND = data["rnd"]
    SAMPLE = data["sample"]
    USE_VIDEO = data["use_video"]
    USE_SPCAM_CAMERA = data["spcam"]

    del args["left_embodiment_config"]["static_camera_list"][1]
    del args["right_embodiment_config"]["static_camera_list"][1]

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        with open(temp_restore_path, "wb") as f:
            pickle.dump(args["eval_video_save_dir"], f)
            pickle.dump(TASK_ENV.test_num, f)
            pickle.dump(suc_list, f)

        if USE_SPCAM_CAMERA:
            spcam_cfg = _camera_config["Spcam_OBS"]
            for cam in spcam_cfg.get("cameras", []):
                args["left_embodiment_config"]["static_camera_list"].append({
                    "name": cam["name"],
                    "type": "Spcam_OBS",
                    "position": cam["position"],
                    "forward": cam["forward"],
                    "left": cam["left"],
                })

        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=now_seed, is_test=True, silent_actor_log=True, **args
                )
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except NotImplementedError as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(e)
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq
        TASK_ENV.setup_demo(
            now_ep_num=now_id, seed=now_seed, is_test=True, silent_actor_log=True, **args
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(
            args["task_name"], episode_info_list, test_num
        )
        if RND:
            instruction = np.random.choice(results[0][instruction_type])
            with open(f"./eval_data/{task_name}/{SAMPLE}/{now_id}_inst.pkl", "wb") as f:
                pickle.dump(instruction, f)
        else:
            with open(f"./eval_data/{task_name}/{SAMPLE}/{now_id}_inst.pkl", "rb") as f:
                instruction = pickle.load(f)

        try:
            from openpi.policies.policy import print_episode_section, print_episode_end
        except ImportError:
            print_episode_section = None
            print_episode_end = None
        if print_episode_section is not None:
            print_episode_section(now_id, instruction)
        else:
            print(f"Load Actor {now_id}")
            print(f"Load instruction {now_id}")

        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction
        if USE_VIDEO:
            if TASK_ENV.eval_video_path is not None:
                out_path = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
                ffmpeg = _make_ffmpeg(video_size, out_path, scale_hd=hd) if not USE_SPCAM_CAMERA else None
                ffmpeg_spcam = _make_ffmpeg(video_size, out_path, scale_hd=hd) if USE_SPCAM_CAMERA else None
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg, ffmpeg_spcam)
        else:
            TASK_ENV._set_eval_video_ffmpeg()

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation, TASK_ENV.take_action_cnt == 0)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            suc_list.append(TASK_ENV.test_num)
        else:
            pass

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        if print_episode_end is not None:
            print_episode_end(
                success=succ,
                step=TASK_ENV.take_action_cnt,
                step_lim=TASK_ENV.step_lim,
                task_name=task_name,
                policy_name=args["policy_name"],
                task_config=args["task_config"],
                ckpt_setting=args["ckpt_setting"],
                suc=TASK_ENV.suc,
                test_num=TASK_ENV.test_num,
                seed=now_seed,
            )
        else:
            result_line = "SUCCESS" if succ else "FAIL"
            result_color = "\033[92m" if succ else "\033[91m"
            reset = "\033[0m"
            success_rate = round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)
            print()
            print("=" * 62)
            print(f"  EPISODE RESULT : {result_color}{result_line}{reset}")
            print(f"  Task           : {task_name} | {args['policy_name']} | {args['task_config']} | {args['ckpt_setting']}")
            print(
                f"  Success Rate   : \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}{reset} => "
                f"\033[95m{success_rate}%{reset}, seed: \033[90m{now_seed}{reset}"
            )
            print("=" * 62)
            print()
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, suc_list


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
