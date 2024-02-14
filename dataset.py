import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='../data/*', get_subdirs=True, max_ctx_length=32, data_aug=False):
        print("Loading dataset...")
        self.data_aug = data_aug
        self.data = self.extract_data(directory, get_subdirs)
        self.max_ctx_length = max_ctx_length

    def extract_data(self, directory, get_subdirs):
        data = glob.glob(directory)
        if get_subdirs:
            all_files = []
            for d in data:
                file_data = [os.path.join(d, f"{str(i).zfill(7)}.jpg") for i in range(1, len(os.listdir(d))+1)]
                all_files.extend(file_data)
            return all_files
        else:
            return data

    def __len__(self):
        return len(self.data) - self.max_ctx_length - 1

    def __getitem__(self, key):
        try:
            frame_start = key
            frames = []

            for i_frame in range(frame_start, frame_start + self.max_ctx_length):
                frame_path = self.data[i_frame]
                frame = read_image(frame_path, mode=ImageReadMode.RGB).float() / 255
                frames.append(frame)

        except RuntimeError:
            print("Could not read file", self.data[i_frame])
            return self.__getitem__(random.randint(0, len(self)))

        return torch.stack(frames, dim=-1)

if __name__ == "__main__":
    dataset = Dataset()
    print(len(dataset))
    print(dataset.__getitem__(8)[0].shape)
