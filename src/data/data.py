import os
from typing import List, Any, Tuple, Optional
import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL
import random
from datasets import load_dataset

class SequencesDataset(Dataset):
    def __init__(
        self,
        dataset,
        seq_length: int,
        game_length: int = 1_000,
        transform: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.sequences: List[Tuple[List[int], List[int]]] = []
        self.transform = transform
        self.game_length = game_length
        self.seq_length = seq_length

        self.dataset_indices = [0] + [game_length * i for i in range(1, len(self.dataset) // game_length)]

        if seed is not None:
            random.seed(seed)
        
    @staticmethod
    def get_np_img(tensor: torch.Tensor) -> np.ndarray:
        return (tensor * 127.5 + 127.5).long().clip(0,255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    def get_session(self, index: int, length: int):
        game_index = self.dataset_indices[index]
        index_range = (game_index, game_index + self.game_length)

        start = random.randint(index_range[0] + length, index_range[1] - length)
        end = start + length

        if end > index_range[1]:
            start = index_range[1] - length
            end = index_range[1]
        if start < index_range[0]:
            start = index_range[0]
            end = start + length
        
        return start, end
    
    def get_images(self, start: int, end: int) -> List[torch.Tensor]:
        imgs = [self.transform(self.dataset[i]['image']) for i in range(start, end)]
        return imgs

    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, end = self.get_session(index, self.seq_length + 1)
        actions1 = self.dataset[start:end]['first_0']
        actions2 = self.dataset[start:end]['second_0']
        
        imgs = self.get_images(start, end)

        last_img = imgs[-1]
        actions1 = torch.tensor(actions1)[:-1]
        actions2 = torch.tensor(actions2)[:-1]

        return (last_img, torch.stack(imgs[:-1]), torch.stack([actions1, actions2], dim=0))


if __name__ == "__main__":
    dataset = load_dataset("betteracs/boxing_atari_diffusion")
    transform_to_tensor = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])
    train_dataset = SequencesDataset(dataset, seq_length=8, game_length=1_000, transform=transform_to_tensor)
    print(len(train_dataset))
