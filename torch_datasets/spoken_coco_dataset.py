import json
from pathlib import Path
from typing import Callable, List, Optional, Set

import soundfile as sf
import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset

from torch_datasets.transforms import Audio_TrainTransform, Img_TrainTransform


class SpokenCocoDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        audio_dir: Path,
        ann_file: Path,
        image_ids: Optional[Set[int]] = None,
        img_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
    ) -> None:
        self.img_transform = img_transform
        self.audio_transform = audio_transform
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        self.data: List[tuple[Path, Path, str]] = []
        for record in ann["data"]:
            image_id = int(Path(record["image"]).stem.split("_")[-1])
            if image_ids is not None and image_id not in image_ids:
                continue

            image_name = Path(record["image"])
            if image_name.parts[0] == "val2014" or image_name.parts[0] == "train2014" or image_name.parts[0] == "test2014":
                image_name = Path(*image_name.parts[1:])

            img_path = img_dir / image_name
            for cap in record["captions"]:
                wav_name = Path(cap["wav"])
                if wav_name.parts[0] == "wavs":
                    wav_name = Path(*wav_name.parts[1:])

                if wav_name.parts and wav_name.parts[0] == "val" or wav_name.parts[0] == "train" or wav_name.parts[0] == "test":
                    wav_name = Path(*wav_name.parts[1:])

                self.data.append((img_path, audio_dir / wav_name, cap["text"]))


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_path, wav_path, text = self.data[idx]

        img = io.read_image(str(img_path), mode=io.ImageReadMode.RGB).float() / 255.0
        if self.img_transform is not None:
            img = self.img_transform(img)

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio).T.contiguous()
        if self.audio_transform is not None:
            result = self.audio_transform(audio)
            audio = result[0] if isinstance(result, tuple) else result
        audio = torch.as_tensor(audio, dtype=torch.float32)

        return img, audio, text

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / ".." / "data" / "spokencoco").resolve()

if __name__ == "__main__":
    dataset = SpokenCocoDataset(
        img_dir=DATA_DIR / "val2014",
        audio_dir=DATA_DIR / "wavs/val",
        ann_file=DATA_DIR / "SpokenCOCO_val.json",
        img_transform=Img_TrainTransform(),  
        audio_transform=Audio_TrainTransform()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )