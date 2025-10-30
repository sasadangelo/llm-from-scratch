from torch.utils.data import DataLoader
from simple_dataset import SimpleDataset


class SimpleDataLoader(DataLoader):
    def __init__(
        self,
        text,
        tokenizer,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        dropLast=True,
        num_workers=0,
    ):
        self.tokenizer = tokenizer
        self.dataset = SimpleDataset(text, tokenizer, max_length, stride)
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=dropLast,
            num_workers=num_workers,
        )
