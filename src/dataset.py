from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import openslide 
import pandas as pd
import torch
import sqlite3
import os

from numpy.random import randint, choice
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path


Coords = Tuple[int, int]


class SlideObject:
    """Class to handle WSI objects.
    
    Args:
        slide_path (str, optional): Path to file. Defaults to None.
        annotations (pd.DataFrame, optional): Annotations [e.g. x, y, class]. Defaults to None.
        patch_size (int, optional): Patch size. Defaults to 128.
        level (int, optional): Level to sample patches. Defaults to 0.
        active_map (np.ndarray, optional): Binary Tissue Segmentation Mask. Defaults to None.
        active_map_ds (int, optional): Downsample factor of active map. Defaults to None.

    Raises:
            ValueError: If no filepath is provided.
    """
    def __init__(
        self,
        slide_path: str = None,
        annotations: pd.DataFrame = None,
        patch_size: int = 128,
        level: int = 0,
        active_map: np.ndarray = None,
        active_map_ds: int = None,
        ) -> None:

        if slide_path is None: 
            raise ValueError('Must provide filename')

        self.slide_path = slide_path
        self.annotations = annotations
        self.patch_size = patch_size
        self.level = level
        
        if active_map is not None:
            self.active_map = active_map
            self.active_map_ds = active_map_ds

        self.slide = openslide.open_slide(self.slide_path)

    @property
    def slide_dims(self) -> Coords:
        return self.slide.level_dimensions[self.level]

    @property
    def patch_dims(self) -> Coords:
        return (self.patch_size, self.patch_size)


    def __str__(self) -> str:
        return f"SlideObject for {self.slide_path}"
    

    def load_image(self, coords: Coords) -> np.ndarray:
        """Returns a patch of the slide at the given coordinates."""
        size = (self.patch_size, self.patch_size)
        patch = self.slide.read_region(coords, self.level, size).convert('RGB')
        return patch


    def get_label(self, x: int, y: int) -> int:
        """Whether a patch contains a mitotic figure."""
        mfs = self.annotations.query('label == 1')
        idxs = (mfs.x > x) & (mfs.x < (x + self.patch_size)) \
            & (mfs.y > y) & (mfs.y < (y + self.patch_size))
        if (np.count_nonzero(idxs) > 0):
            return 1
        else:
            return 0
        
    def get_tissue_coverage(self, x: int, y: int) -> float:
        """Returns the tissue coverage of a patch."""
        if self.active_map is None:
            raise ValueError('No active map provided.')
        else:
            x = x // self.active_map_ds
            y = y // self.active_map_ds
            return np.sum(self.active_map[y:y+self.patch_size//self.active_map_ds, x:x+self.patch_size//self.active_map_ds]) / ((self.patch_size //self.active_map_ds)**2)
    
    def get_thumbnail(self, size: int = 256) -> np.ndarray:
        """Returns a thumbnail of the slide."""
        return np.array(self.slide.get_thumbnail((size, size)))
    
    def get_boxes(self,x:int,y:int,width:int,height:int,box_size:Tuple[int,int] = (50,50)):
        """Returns the annotations in a given patch"""
        xmax = x + width
        ymax = y + height
        # load all annotations in patch and convert to numpy
        mf = self.annotations.query(f'x >= {x} and x <= {xmax} and y >= {y} and y <= {ymax} and label == 1')[['x', 'y']].to_numpy()
        if len(mf) > 0:
            boxes = np.zeros((len(mf),4), dtype=int)
            labels = np.ones(len(mf), dtype=int)
            # convert the centers in mf into bounding boxes and bring them to the patch coordinate system
            for i in range(len(mf)):
                x_center, y_center = mf[i]
                x1 = x_center - box_size[0] // 2 - x if x_center - box_size[0] // 2 - x >= 0 else 0
                y1 = y_center - box_size[1] // 2 - y if y_center - box_size[1] // 2 - y >= 0 else 0
                x2 = x_center + box_size[0] // 2 - x if x_center + box_size[0] // 2 - x <= width else width
                y2 = y_center + box_size[1] // 2 - y if y_center + box_size[1] // 2 - y <= height else height
                boxes[i] = [x1,y1,x2,y2]
                
        else:
            boxes = np.zeros((0, 4), dtype=np.int32)
            labels = np.zeros(1, dtype=np.int32)
                
        
        return boxes, labels
        

class Mitosis_Base_Dataset(Dataset):
    """Base dataset class for mitosis classificaiton.

    Contains functionality to load SlideObjects. Functionality for sampling 
    patches needs to be implemented with `sample_patches`, `sample_func`.

    Args:
        csv_file (str): Path to database or Pandas Dataframe.
        image_dir (str): Directory with images. 
        indices (np.array, optinal): Indices to select slides from database. Defaults to None.
    """
    def __init__(
        self,
        csv_file: Union[str, pd.DataFrame],
        image_dir: str
        ) -> None:
        # check whether csv_file is a path or a csv file or a sql database
        if isinstance(csv_file, str) or isinstance(csv_file, Path):
            if not os.path.exists(csv_file):
                raise ValueError('Need to provide csv file.')
            else:
                self.csv_file = csv_file
                self.data = self.load_database()
        # check whether csv_file is a pandas dataframe
        elif isinstance(csv_file, pd.DataFrame):        
            self.csv_file =csv_file
            self.data = csv_file
        else:
            raise ValueError('Need to provide a path to a valid csv file or a pandas dataframe.')
        
        # check if only 0 and 1 are in the labels 
        assert sorted(self.data.label.unique().tolist()) == [0,1], 'Labels need to be 0 and 1'
        
        # check image directory
        if not os.path.isdir(image_dir):
            raise ValueError('Need to provide directory with images.')
        else:
            self.image_dir = image_dir


    def load_database(self) -> pd.DataFrame:
        """Loads database from csv file."""
        return pd.read_csv(self.csv_file)


    def return_split(self, split:str = 'train', **kwargs):
        """ Returns either training, validation or test set.

        Args:
            indices (np.array): Indices to select for split.
            training (bool, optional): Whether to use train or val dataset. Defaults to False.
        """
        if split == 'train' or split == 'val':
            return Mitosis_Dataset(split = split,
                                            csv_file = self.csv_file,
                                            image_dir = self.image_dir,
                                            **kwargs)
        elif split == 'test':
            return Mitosis_Test_Dataset(split = split,
                                              csv_file = self.csv_file,
                                              image_dir = self.image_dir,
                                              **kwargs)
        else:
            raise ValueError(f'Unknown split {split}.')
        

    def load_slide_objects(self):
        """Function to initialize slide objects from dataframe."""
        raise NotImplementedError


    def sample_patches(self):
        """Sample patches for each epoch."""
        raise NotImplementedError


    def sample_func(self):
        """Sample coordinates for a single patch."""
        raise NotImplementedError


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None


    def get_ids(self):
        return self.data['slide'].unique()


    def get_labels(self):
        return self.data['label'].unique()


    def summarize(self):
        print('\nNumber of slides: {}'.format(len(self.data['slide'].unique())))
        print('Number of mitosis: {}'.format(len(self.data.query('label == 1'))))
        print('Number of imposter: {}'.format(len(self.data.query('label == 0'))))


    @staticmethod
    def collate_fn(batch):
        """Collate function for the data loader."""
        images = list()
        targets = list()
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.int64)
        return images, targets



class Mitosis_Dataset(Mitosis_Base_Dataset):
    """Datasat for mitosis classification.

    Randomly samples patches around mitotic figures or imposters.
    Patch coordinsates are slightly shifted after sampling to add more variability.

    This class is used for training and validation. The test dataset should be constructed 
    from `Mitosis_Test_Dataset`.

    Args:
        csv_file (str): Path to database. 
        image_dir (str): Directory with images. 
        pseudo_epoch_length (int, optional): Number of patches for each epoch. Defaults to 512.
        mit_prob (float, optional): Percentage of patches with mitotic figures. Defaults to 0.5.
        patch_size (int, optional): Patch size. Defaults to 128.
        level (int, optional): Level to sample. Defaults to 0.
        transforms (Union[List[Callable], Callable], optional): Transformations. Defaults to None.
    """
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        split: str = 'train',
        pseudo_epoch_length: int = 512,
        mit_prob: float = 0.5,
        offset_scale: float = 0.1,
        patch_size: int = 128, 
        level: int = 0,
        transforms: Union[List[Callable], Callable] = None
        ) -> None:
        Mitosis_Base_Dataset.__init__(self, csv_file=csv_file, image_dir=image_dir)

        self.split = split
        self.pseudo_epoch_length = pseudo_epoch_length
        self.mit_prob = mit_prob
        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms
        self.offset_scale = offset_scale

        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()


    def load_slide_objects(self) -> Dict[str, SlideObject]:
        """Initializes slide objects from dataframe.

        Returns:
            Dict[str, SlideObject]: Dictionary with all slide objects. 
        """
        fns = self.data.query(f'split == "{self.split}"').filename.unique().tolist()
        
        slide_objects = {}
        for fn in tqdm(fns, desc='Initializing slide objects'):
            slide_path = os.path.join(self.image_dir, fn)
            annotations = self.data.query(f'filename == @fn and split == "{self.split}"')[['x', 'y', 'label']].reset_index()
            slide_objects[fn] = SlideObject(
                slide_path=slide_path,
                annotations=annotations,
                patch_size=self.patch_size,
                level=self.level
            )
        return slide_objects


    def sample_patches(self) -> Dict[str, Dict[str, Coords]]:
        """Samples patches from all slides with equal probability.
        Proportions of patches with mitotic figures or imposters can be adjusted with mit_prob.

        Returns:
            Dict[str, Dict[str, Coords]]: Dictionary with {idx: coords, label}
        """
        # sample slides
        slides = choice(list(self.slide_objects.keys()), size=self.pseudo_epoch_length, replace=True)
        # sample patches
        patches = {}
        for idx, slide in enumerate(slides):
            patches[idx] = self.sample_func(slide, self.mit_prob)
        return patches


    def resample_patches(self):
        """Loads a new set of patches."""
        self.samples = self.sample_patches()
        print('Sampled new patches!')


    def sample_func(
        self,
        fn: str, 
        mit_prob: float,
        ) -> Dict[str, Tuple[Coords, int]]:
        """Samples patches randomly from a slide.

        Labels are transformed into a binary problem [0=no mitosis, 1=mitosis].

        Args:
            fn (str): Filename (e.g. "042.tiff")
            mit_prob (float): Proportion of patches with mitotic figure. 1-mit_prob is the proportion of imposters.

        Returns:
            Dict[str, Tuple[Coords, int]]: Dictionary with filenames, patch coordinates and the label.
        """
        # get slide object
        sl = self.slide_objects[fn]

        # get dims
        slide_width, slide_height = sl.slide_dims
        patch_width, patch_height = sl.patch_dims

        # create sampling probabilites
        sample_prob = np.array([mit_prob, 1-mit_prob])

        # sample case from probabilites (0 = mitosis, 1 = imposter)
        case = choice(2, p=sample_prob)
        
        if case == 0:     
            # filter mitosis cases
            mask = sl.annotations.label == 1

            if np.count_nonzero(mask) == 0:
                # no mitosis available -> sample imposter
                case = 1
            else:       
                # get annotations
                MF = sl.annotations[['x', 'y']][mask]

                # sample mitosis
                idx = randint(MF.shape[0])
                x, y = MF.iloc[idx]

        if case == 1:
            # sample imposter
            mask = sl.annotations.label == 0

            if np.count_nonzero(mask) == 0:
                # no imposter available -> random patch
                x = randint(patch_width / 2, slide_width-patch_width / 2)
                y = randint(patch_height / 2, slide_height-patch_height / 2)

            else:
                # get annotations
                NMF = sl.annotations[['x', 'y']][mask]
                # sample imposter
                idx = randint(NMF.shape[0])
                x, y = NMF.iloc[idx]

        # set offsets
        xoffset = randint(-patch_width, patch_width) * self.offset_scale
        yoffset = randint(-patch_height, patch_height) * self.offset_scale

        # shift coordinates and return top left corner
        x = int(x - patch_width / 2 + xoffset) 
        y = int(y - patch_height / 2 + yoffset)

        # avoid black borders
        if x + patch_width > slide_width:
            x = slide_width - patch_width
        elif x < 0:
            x = 0
        
        if y + patch_height > slide_height:
            y = slide_height - patch_height
        elif y < 0:
            y = 0

        label = sl.get_label(x, y)

        return {'file': fn, 'coords': (x, y), 'label': label}


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        file, coords, label = sample['file'], sample['coords'], sample['label']
        slide = self.slide_objects[file]

        img = slide.load_image(coords)
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(np.array(img) / 255).permute(2, 0, 1).type(torch.float32)

        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label, file, coords


class Mitosis_Test_Dataset(Mitosis_Dataset):
    """Dataset for mitosis classification. 

    Used for evaluation of a classifier. Instead of sampling patches randomly,
    all annotations from the dataset are loaded and centered around the annotation.

    Args:
        csv_file (str): Path to database.      
        image_dir (str): Directory with images. 
        patch_size (int, optional): _description_. Defaults to 128.
        level (int, optional): _description_. Defaults to 0.
        transforms (Union[List[Callable], Callable], optional): _description_. Defaults to None.
        score_threshold (float, optional): Threshold for MF-Candidates to return, only effective as used for second stage with MF_Candidates. Defaults to None.
    """
    def __init__(
        self, 
        csv_file: str,
        image_dir: str,
        split: str = 'test',
        patch_size: int = 128, 
        level: int = 0,
        only_mitosis: bool = False,
        transforms: Union[List[Callable], Callable] = None) -> None:
        Mitosis_Base_Dataset.__init__(self, csv_file, image_dir)

        self.patch_size = patch_size
        self.level = level
        self.transforms = transforms
        self.split = split

        # select only the data of the respective split
        self.data = self.data.query(f'split == "{split}"')
        if only_mitosis:
            self.data = self.data.query(f'split == "{split}" and label==1')

        self.slide_objects = self.load_slide_objects()
        self.samples = self.sample_patches()
        

    def sample_patches(self) -> Dict[str, Dict[str, Coords]]:
        """
        Samples patches from the dataset.

        Returns:
            patches (Dict[str, Dict[str, Coords]]): A dictionary containing information about each sampled patch.
                Each patch is represented by a unique index (key) and contains the following information:
                - 'file': The filename from which the patch was sampled.
                - 'coords': The coordinates (x, y) of the patch's top-left corner.
                - 'label': The label of the patch (1 for positive, 0 for negative).
        """
        patches = {}
        for idx, (_, row) in enumerate(self.data.iterrows()):
            fn, x, y = row.filename, int(row.x - self.patch_size / 2), int(row.y - self.patch_size / 2)
            label = row.label
            patches[idx] = {'file': fn, 'coords': (x, y), 'label': label}
        return patches
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        file, coords, label = sample['file'], sample['coords'], sample['label']
        slide = self.slide_objects[file]

        try:
            img = slide.load_image(coords)
        except OSError as e:
            print(f"[ERROR] Failed to load image for idx={idx}, slide={file}, coords={coords}: {e}")
            raise
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(np.array(img) / 255).permute(2, 0, 1).type(torch.float32)

        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label, file, coords

