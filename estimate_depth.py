"""Compute depth maps for images in the input folder.
"""
import argparse
import os
import glob
import torch
import utils
import cv2
import re
import sys

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet


def run(img_paths, depth_paths, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    # get input
    num_images = len(img_paths)

    print("start processing {} images".format(num_images))

    for ind, (img_name, depth_name) in enumerate(zip(img_paths, depth_paths)):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # create output folder
        os.makedirs(os.path.dirname(depth_name), exist_ok=True)

        # output
        utils.write_depth(depth_name, prediction, bits=2)

    print("finished")


def cvt_format(path, regex, out_format):
    mo = regex.search(path)
    assert len(mo.groups()) == 1

    key = os.path.splitext(mo.group(1))[0]
    return out_format.format(key)


def main(args):
    in_paths = glob.glob(args.in_format)
    regex = re.compile(args.in_format.replace('*', '(.*)'))
    out_paths = [cvt_format(p, regex, args.out_format) for p in in_paths]
    print(in_paths, out_paths)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(in_paths, out_paths, args.model_path)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_format', help="input filename format")
    parser.add_argument('--out_format', help="no extension is needed. e.g., out_dir/{}")
    parser.add_argument('--model_path', default='model.pt')
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
