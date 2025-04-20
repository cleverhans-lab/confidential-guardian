import os
import torch
import pickle
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms


class CIFAR100WithCoarseLabels(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100WithCoarseLabels, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        # Initialize lists for coarse targets
        self.coarse_targets = []
        
        # Path to the CIFAR-100 data
        cifar100_dir = os.path.join(root, self.base_folder)
        
        # Load the data file
        if self.train:
            file = os.path.join(cifar100_dir, 'train')
        else:
            file = os.path.join(cifar100_dir, 'test')
        
        with open(file, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['fine_labels']  # Fine labels
            self.coarse_targets = entry['coarse_labels']  # Coarse labels
        
        # Load meta data to get label names
        with open(os.path.join(cifar100_dir, 'meta'), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
            self.fine_label_names = meta['fine_label_names']
            self.coarse_label_names = meta['coarse_label_names']
        
        # Decode label names from bytes to strings
        self.fine_label_names = [label.decode('utf-8') if isinstance(label, bytes) else label for label in self.fine_label_names]
        self.coarse_label_names = [label.decode('utf-8') if isinstance(label, bytes) else label for label in self.coarse_label_names]

    def __getitem__(self, index):
        """
        Override the __getitem__ method to return both fine and coarse labels.
        """
        img, fine_label, coarse_label = self.data[index], self.targets[index], self.coarse_targets[index]
        img = torchvision.transforms.functional.to_pil_image(img.reshape(3, 32, 32).transpose(1, 2, 0))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            fine_label = self.target_transform(fine_label)
            coarse_label = self.target_transform(coarse_label)
        
        return img, fine_label, coarse_label



class CIFAR100WithUncertainty(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, uncertain_fine_labels=None):
        """
        Initializes the dataset, including fine labels, coarse labels, and uncertainty indicators.

        Args:
            root (str): Root directory of dataset.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            uncertain_fine_labels (list, optional): List of fine label names to be marked as uncertain.
        """
        super(CIFAR100WithUncertainty, self).__init__(
            root, train=train, transform=transform, target_transform=target_transform, download=download
        )
        
        # Initialize lists for coarse targets
        self.coarse_targets = []
        
        # Path to the CIFAR-100 data
        cifar100_dir = os.path.join(root, self.base_folder)
        
        # Load the data file
        if self.train:
            file = os.path.join(cifar100_dir, 'train')
        else:
            file = os.path.join(cifar100_dir, 'test')
        
        with open(file, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['fine_labels']  # Fine labels
            self.coarse_targets = entry['coarse_labels']  # Coarse labels
        
        # Load meta data to get label names
        with open(os.path.join(cifar100_dir, 'meta'), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
            self.fine_label_names = meta['fine_label_names']
            self.coarse_label_names = meta['coarse_label_names']
        
        # Decode label names from bytes to strings if necessary
        self.fine_label_names = [label.decode('utf-8') if isinstance(label, bytes) else label for label in self.fine_label_names]
        self.coarse_label_names = [label.decode('utf-8') if isinstance(label, bytes) else label for label in self.coarse_label_names]
        
        # Determine the indices of uncertain fine labels
        if uncertain_fine_labels is not None:
            self.uncertain_fine_label_indices = [self.fine_label_names.index(label) for label in uncertain_fine_labels if label in self.fine_label_names]
        else:
            self.uncertain_fine_label_indices = []
    
    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return image, fine label, coarse label, and uncertainty indicator.
        """
        img, fine_label, coarse_label = self.data[index], self.targets[index], self.coarse_targets[index]
        img = torchvision.transforms.functional.to_pil_image(img.reshape(3, 32, 32).transpose(1, 2, 0))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            fine_label = self.target_transform(fine_label)
            coarse_label = self.target_transform(coarse_label)
        
        # Determine the uncertainty indicator
        uncertainty = 1 if fine_label in self.uncertain_fine_label_indices else 0
        
        return img, fine_label, coarse_label, uncertainty



num_bins = 12
age_min = 0
age_max = 116
bins = np.linspace(age_min, age_max, num_bins + 1)  # 13 edges for 12 bins

def assign_age_bin(age):
    """
    Assigns an age to a bin index.
    """
    bin_index = np.digitize(age, bins) - 1
    bin_index = np.clip(bin_index, 0, num_bins - 1)
    return bin_index

def parse_utkface_filename(filename):
    """
    Parses the UTKFace filename to extract age, gender, and race.
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) < 4:
        raise ValueError(f"Filename {filename} does not conform to expected format.")
    
    age = int(parts[0])
    gender = int(parts[1])  # 0: Male, 1: Female
    race = int(parts[2])    # 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
    return age, gender, race

class UTKFaceDatasetMultiTask(Dataset):
    def __init__(self, root_dir, transform=None, target_type='age_class', num_bins=12):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_type (string): Type of label to return ('age_class', 'gender', 'race').
            num_bins (int): Number of bins for age classification.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_type = target_type
        self.num_bins = num_bins
        self.image_files = [
            os.path.join(root_dir, file) for file in os.listdir(root_dir) 
            if file.endswith('.jpg') and len(file.split('_')) >= 4
        ]
        self.bins = np.linspace(age_min, age_max, num_bins + 1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        age, gender, race = parse_utkface_filename(img_path)
        
        # Assign age to a bin
        age_class = assign_age_bin(age)
        
        # Define the uncertainty indicator
        uncertainty_indicator = 1 if gender == 0 and race == 0 else 0
        # uncertainty_indicator = 1 if gender == 1 else 0 #and race == 0 else 0
        # uncertainty_indicator = 1 if race == 2 else 0 
        
        if self.target_type == 'age_class':
            primary_label = age_class
        elif self.target_type == 'gender':
            primary_label = gender
        elif self.target_type == 'race':
            primary_label = race
        else:
            raise ValueError("target_type must be 'age_class', 'gender', or 'race'")
        
        if self.transform:
            image = self.transform(image)
        
        return image, primary_label, uncertainty_indicator