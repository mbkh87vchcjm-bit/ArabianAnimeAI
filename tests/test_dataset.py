import torch
from data.dataset import BaseDataset

class DummyTestDataset(BaseDataset):
    def __init__(self, num_items=10):
        self.num_items = num_items

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> dict:
        return {"index": index, "tensor": torch.tensor([index, index + 1], dtype=torch.float32)}

def test_base_dataset_len_and_getitem():
    dataset = DummyTestDataset(num_items=5)
    assert len(dataset) == 5

    item = dataset[2]
    assert item["index"] == 2
    assert torch.equal(item["tensor"], torch.tensor([2.0, 3.0]))
