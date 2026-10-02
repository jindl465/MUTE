import os

import torch
import torchvision
from colorama import Fore, Style


def cprint(message, color):
    color_code = {
        "r": Fore.RED,
        "g": Fore.GREEN,
        "y": Fore.YELLOW,
        "b": Fore.BLUE,
    }.get(color, "")
    print(f"{color_code}{message}{Style.RESET_ALL}")


def mp(path):
    if not path:
        return
    directory = path if path.endswith(os.sep) else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def si(tensor, path, to_01=False, normalize=False):
    if to_01:
        torchvision.utils.save_image((tensor + 1) / 2, path, normalize=normalize)
    else:
        torchvision.utils.save_image(tensor, path, normalize=normalize)
