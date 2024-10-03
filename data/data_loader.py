import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

class MultiFileHSIDataLoader:
    def __init__(self, param: Dict):
        self.data_param = param['data']
        self.data_path_prefix = self.data_param.get('data_path_prefix', 'data/dataset')
        self.data_sign = self.data_param.get('data_sign', 'Plastic')
        self.patch_size = self.data_param.get('patch_size', 5)  
        self.margin = (self.patch_size - 1) // 2   #For constructing patchwindow for corner pixels of image, utilized in padding
        self.batch_size = self.data_param.get('batch_size', 65) 
        self.spectral_size = self.data_param.get("spectral_size", 0)
        self.norm_type = self.data_param.get("norm_type", 'max_min')
        self.padding = self.data_param.get("padding", False)
        self.pca_components = self.data_param.get('pca', 0)
        self.num_classes = self.data_param.get('num_classes', 13)
        if self.data_sign == 'Plastic':
            self.matlab_files = self.data_param.get('matlab_files', [])

        self.images = []
        self.labels = []
        self.index2pos_train = []
        self.index2pos_valid = []
        self.index2pos_test = []

    def load_data_from_matlab(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from MATLAB file."""
        data = sio.loadmat(file_path)
        if self.data_sign in ['IndianPine', 'Houston', 'Pavia']:
            img = data['input']
            TR = data['TR']
            VA = data['TE']
            TE = data['TE']
            labels = TR + TE
        else:
            img = data['HSI_data']
            TR = data['TR']
            VA = data['VA']
            TE = data['TE']
            labels = TR + VA + TE   #if we didn't have combined Ground Truth, we can get it like this
        return img, labels, TR, VA, TE
    
    def load_raw_data(self):
        """Load raw data from MATLAB files."""
        if self.data_sign in ['IndianPine', 'Houston', 'Pavia']:
            file_path = f'{self.data_path_prefix}/{self.data_sign}/{self.data_sign}.mat'
            img, labels, TR, VA, TE = self.load_data_from_matlab(file_path)
            self.images.append(img)
            self.labels.append(labels)
            self.index2pos_train.append(self.extract_indices(TR))
            self.index2pos_valid.append(self.extract_indices(VA))
            self.index2pos_test.append(self.extract_indices(TE))
        else:
            for file_name in self.matlab_files:
                file_path = f'{self.data_path_prefix}/{self.data_sign}/{file_name}'
                img, labels, TR, VA, TE = self.load_data_from_matlab(file_path)
                self.images.append(img)
                self.labels.append(labels)
                self.index2pos_train.append(self.extract_indices(TR))
                self.index2pos_valid.append(self.extract_indices(VA))
                self.index2pos_test.append(self.extract_indices(TE))

    def extract_indices(self, label_mask: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """Extract indices from label mask."""
        indices = {}
        index = 0
        for i in range(label_mask.shape[0]):
            for j in range(label_mask.shape[1]):
                if label_mask[i, j] > 0:
                    indices[index] = (i, j)
                    index += 1
        return indices
    
    def _padding(self, X):
        w, h, c = X.shape
        padded_img = np.zeros((w + 2 * self.margin, h + 2 * self.margin, c))
        padded_img[self.margin:w + self.margin, self.margin:h + self.margin, :] = X
        return padded_img.astype(np.float32)
    
    def apply_pca(self, X):
        """Apply PCA to the input data."""
        flattened = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=self.pca_components, whiten=True)
        transformed = pca.fit_transform(flattened)
        transformed = np.reshape(transformed, (X.shape[0], X.shape[1], self.pca_components))
        return transformed
    
    def calculate_global_min_max(self, images):
        """
        Calculate global min and max values for each band across all images.
        """
        num_bands = images[0].shape[2]
        global_min = np.full(num_bands, np.inf)
        global_max = np.full(num_bands, -np.inf)

        # Calculate global min and max across all images with respect to bands
        for img in images:
            for i in range(num_bands):
                band_slice = img[:, :, i]
                global_min[i] = min(global_min[i], np.min(band_slice[band_slice != 0]))  # ignore the 0 values that I padded along the final row & column to make the image dimensions same 
                global_max[i] = max(global_max[i], np.max(band_slice[band_slice != 0]))  

        return global_min, global_max

    def normalize_data(self, data: np.ndarray, global_min: np.ndarray = None, global_max: np.ndarray = None) -> np.ndarray:
        """Normalize data based on specified normalization type."""
        if self.norm_type == 'max_min' and global_min is not None and global_max is not None:
            num_bands = data.shape[2]
            norm_data = np.zeros_like(data, dtype=np.float32)
            for i in range(num_bands):
                norm_data[:, :, i] = (data[:, :, i] - global_min[i]) / (global_max[i] - global_min[i])
                
        elif self.norm_type == 'max_min' and (global_min is None or global_max is None):
            num_bands = data.shape[2]
            norm_data = np.zeros_like(data, dtype=np.float32)
            for i in range(num_bands):
                norm_data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / (np.max(data[:, :, i]) - np.min(data[:, :, i]))
                
        else:
            norm_data = data
        return norm_data.astype(np.float32)

    def prepare_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], List[Dict], List[Dict]]:
        """Prepare data for training and testing."""
        self.load_raw_data()
        norm_images = []
        if self.data_sign == 'Plastic':
            global_min, global_max = self.calculate_global_min_max(self.images)
        else:
            global_min, global_max = None, None
        
        for img in self.images:
            norm_img = self.normalize_data(img, global_min, global_max)
            if self.pca_components > 0:
                norm_img = self.applyPCA(norm_img)
            if self.spectral_size > 0:
                norm_img = norm_img[:, :, :self.spectral_size]
            if self.padding == True:                            # Padding not needed for plastic data, Image corner pixels are background pixels so not selected
                norm_img = self._padding(norm_img)
            norm_images.append(norm_img)
        return norm_images, self.labels, self.index2pos_train, self.index2pos_valid, self.index2pos_test

    def generate_torch_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generate PyTorch DataLoader for train, validation, and test sets."""
        images, labels, train_indices, valid_indices, test_indices = self.prepare_data()

        trainset = MultiImageDataSetIter(images, labels, train_indices, self.margin, self.patch_size, self.data_sign)
        validset = MultiImageDataSetIter(images, labels, valid_indices, self.margin, self.patch_size, self.data_sign)
        testset = MultiImageDataSetIter(images, labels, test_indices, self.margin, self.patch_size, self.data_sign)

        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return train_loader, valid_loader, test_loader

class MultiImageDataSetIter(Dataset):
    def __init__(self, images: List[np.ndarray], labels: List[np.ndarray], index2pos: List[Dict], margin: int, patch_size: int, data_sign: str):
        self.images = images
        self.labels = labels
        self.index2pos = index2pos
        self.size = sum(len(pos) for pos in index2pos)
        self.margin = margin
        self.patch_size = patch_size
        self.data_sign = data_sign

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine the correct image and index within that image and return the patch and label
        for img_idx, img_pos in enumerate(self.index2pos):
            if index < len(img_pos):
                start_x, start_y = img_pos[index]
                image = self.images[img_idx]
                label = self.labels[img_idx]
                break
            index -= len(img_pos)
        
        if self.data_sign in ['IndianPine', 'Houston', 'Pavia']:
            padded_start_x = start_x + self.margin # Since we added margin to the image to extract patches from corners, we need to compensate for that    
            padded_start_y = start_y + self.margin
            
            patch = image[padded_start_x - self.margin:padded_start_x + self.margin + 1, 
                          padded_start_y - self.margin:padded_start_y + self.margin + 1, :]
        
        elif self.data_sign == 'Plastic':
            #In case of Plastic data, we don't need to do any padding, since the image corner pixels are background pixels never selected as training data
            patch = image[start_x - self.margin:start_x + self.margin + 1, 
                          start_y - self.margin:start_y + self.margin + 1, :]


        patch = patch.transpose((2, 0, 1))
        label_value = label[start_x, start_y] - 1
        return torch.FloatTensor(patch.copy()), torch.LongTensor([label_value])[0]

    def __len__(self) -> int:
        return self.size