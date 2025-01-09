import numpy as np
import os
class MNISTDataLoader:
    def __init__(self, dataset: str, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        if not os.path.exists(dataset):
            raise FileNotFoundError(f"Dataset not found in path: {dataset}")
        self.data = np.load(dataset)
        self.data_index = 0
        print(self.data)
    
    def get_validation_accuracy():
        pass

    def train_dataset(self):
        # d = np.column_stack(self.data[self.data_index: ])
        train_data: np.ndarray = self.data['train_images']
        train_label: np.ndarray = self.data['train_labels']
        if (train_data.shape[0] % self.batch_size) != 0:
            raise ValueError(f"Batch size {self.batch_size} is not compatible with dataset size {train_data.shape[0]}")
        reshaped_train_data = train_data.reshape(train_data.shape[0]//self.batch_size, self.batch_size, train_data.shape[1])
        reshaped_train_label = train_label.reshape(train_label.shape[0]//self.batch_size, self.batch_size)
        return reshaped_train_data, reshaped_train_label
    
    def test_dataset(self, test_batch_size: int = 100):
        self.test_batch_size = test_batch_size
        test_data: np.ndarray = self.data['test_images']
        test_label: np.ndarray = self.data['test_labels']
        print(test_data.shape)
        if (test_data.shape[0] % self.test_batch_size) != 0:
            raise ValueError(f"Batch size {self.test_batch_size} is not compatible with dataset size {test_data.shape[0]}")
        reshaped_test_data = test_data.reshape(test_data.shape[0]//self.test_batch_size, self.test_batch_size, test_data.shape[1])
        reshaped_test_label = test_label.reshape(test_label.shape[0]//self.test_batch_size, self.test_batch_size)
        return reshaped_test_data, reshaped_test_label
    def __iter__(self):
        return self.data

dataloader = MNISTDataLoader("./data/mnist.npz", 30)
dataloader.test_dataset()