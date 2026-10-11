from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import v2


class Img_TrainTransform:
    def __init__(
        self,
        size: int = 224,
        crop_scale: Optional[Tuple] = (0.5, 1.0),
        hflip_p: Optional[float] = 0.5,
        jitter_params: Optional[Tuple[float, float, float, float]] = (
            0.4,
            0.4,
            0.2,
            0.1,
        ),
    ) -> None:
        self.size = size
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.jitter_params = jitter_params
        self.transforms_list: list[Callable] = []
        
        if self.crop_scale is not None:
            self.transforms_list.append(
                v2.RandomResizedCrop(size=self.size, scale=self.crop_scale)
            )
        else:
            self.transforms_list.append(v2.Resize(size=self.size))
        
        if self.hflip_p is not None:
            self.transforms_list.append(v2.RandomHorizontalFlip(p=self.hflip_p))
        
        if self.jitter_params is not None:
            self.transforms_list.append(v2.ColorJitter(*self.jitter_params))
        
        self.transforms_list.append(
            v2.Normalize(
                mean=[1, 1, 1], # placeholder
                std=[1, 1, 1] # placeholder
                )
            )        
        self.transform = v2.Compose(self.transforms_list)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)


class Img_ValTransform:
    def __init__(self, size: int = 224, use_ccrop: bool = False) -> None:
        self.size = size
        self.transforms_list: list[Callable] = []
        if use_ccrop:
            self.transforms_list.append(v2.Resize(self.size + 32))
            self.transforms_list.append(v2.CenterCrop(self.size))
        else:
            self.transforms_list.append(v2.Resize((self.size, self.size)))
        self.transforms_list.append(
            v2.Normalize(
                mean=[1, 1, 1], # placeholder
                std=[1, 1, 1], # placeholder
            )
        )
        self.transform = v2.Compose(self.transforms_list)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)

class Audio_TrainTransform:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_sec: float = 10.0, # placeholder
        normalize: bool = True,
    ) -> None:
        self.max_samp = int(sample_rate * max_sec)
        self.normalize = normalize

    def __call__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        n = wav.shape[-1]
        if n > self.max_samp:
            start = torch.randint(0, n - self.max_samp + 1, (1,)).item()
            wav = wav[start : start + self.max_samp]

        length = wav.shape[-1]

        if self.normalize:
            wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        wav = F.pad(wav, (0, self.max_samp - length))
        return wav, length
    
class Audio_ValTransform:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_sec: float = 10.0, # placeholder
        normalize: bool = True,
    ) -> None:
        self.max_samp = int(sample_rate * max_sec)
        self.normalize = normalize

    def __call__(self, wav: torch.Tensor) -> Tuple[torch.Tensor, int]:
        wav = wav[: self.max_samp]
        length = wav.shape[-1]

        if self.normalize:
            wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        wav = F.pad(wav, (0, self.max_samp - length))
        return wav, length