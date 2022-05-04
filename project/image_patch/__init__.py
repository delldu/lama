"""Image/Video Patch Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import math
import torch.nn.functional as F

import redos
import todos

from . import patch

import pdb


def load_model(model, path):
    """Load model."""

    if not os.path.exists(path):
        raise IOError(f"Model checkpoint '{path}' doesn't exist.")

    # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        n = n.replace("generator.", "")
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def get_model():
    """Create model."""

    model_path = "models/image_lama.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = patch.FFCResNetGenerator()
    # todos.model.load(model, checkpoint)
    load_model(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    # model = torch.jit.script(model)

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_lama.torch"):
    #     model.save("output/image_lama.torch")

    return model, device


def pad_tensor(tensor, times=8):
    B, C, H, W = tensor.shape
    Hnew = int(times * math.ceil(H / times))
    Wnew = int(times * math.ceil(W / times))
    Bottom = Hnew - H
    Right = Wnew - W
    PadDim = (0, Right, 0, Bottom)  # l, r, t, b
    return F.pad(tensor, pad=PadDim, mode="reflect")


def model_forward(model, device, input_tensor):
    H, W = input_tensor.size(2), input_tensor.size(3)
    input_tensor = pad_tensor(input_tensor)
    with torch.no_grad():
        output_tensor = todos.model.forward(model, device, input_tensor)
    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.patch(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  patch {input_file} ...")
        try:
            input_tensor = todos.data.load_rgba_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_lama", do_service, host, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_rgba_tensor(filename)
        input_mask = (input_tensor[:, 3:4, :, :] < 0.9).float()
        input_content = input_tensor[:, 0:3, :, :] * (1.0 - input_mask)
        input_tensor = torch.cat((input_content, input_mask), dim=1)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = model_forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor[:, 0:3, :, :], predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  patch {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # keep tensor 1x4xHxW
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.patch(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, host="localhost", port=6379):
    return redos.video.service(name, "video_patch", video_service, host, port)
