import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
# from torchvision.datasets import ImageFolder


class YouTubeVOSLoader(Dataset):
    def __init__(self, root = 'data/youtube_vos', mode = 'train', fraction = 1.0, image_transformation = None, mask_transformation = None, num_frames = 5):
        super(YouTubeVOSLoader, self).__init__()
        self.root = os.path.abspath(root)
        self.mode = mode
        self.fraction = fraction
        self.image_transformation = image_transformation
        self.mask_transformation = mask_transformation

        if self.mode == 'train':
            self.data_dir = os.path.join(self.root, 'train')
        elif self.mode == 'val':
            self.data_dir = os.path.join(self.root, 'valid')

        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.target_dir = os.path.join(self.data_dir, 'Annotations')
        self.filenames = os.listdir(self.image_dir)
        self.num_samples = int(self.fraction * len(self.filenames))

        self.max_sequence_len = num_frames

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, index):

        if self.mode == 'train':
            self.video_name = self.filenames[index]
            self.image_names = os.listdir(os.path.join(self.image_dir, self.video_name))
            self.image_names = sorted(self.image_names)
            self.target_names = os.listdir(os.path.join(self.target_dir, self.video_name))
            self.target_names = sorted(self.target_names)
            self.current_len = len(self.image_names)
            # print(len(self.image_names), len(self.target_names))
            # print(self.video_name)
            # print(self.image_names, self.target_names)

            self.image_name_list = []

            self.video = []
            self.segmented_video = []

            actual_frames_in_video = len(self.image_names)
            num_frames_loaded = 0
            left_to_right = True
            frame_index = -1
            while num_frames_loaded != self.max_sequence_len:
                if left_to_right:
                    frame_index += 1
                else:
                    frame_index -= 1


                filename = os.path.join(self.image_dir, self.video_name, self.image_names[frame_index])
                try:
                    image = Image.open(filename)
                    num_frames_loaded += 1

                    if self.image_transformation:
                        image = self.image_transformation(image)
                    self.video.append(image)
                    self.image_name_list.append(self.image_names[frame_index])
                except:
                    print('IMAGE loading issue', self.video_name)
                
                if frame_index == (actual_frames_in_video - 1):
                    left_to_right = False
                elif frame_index == 0:
                    left_to_right = True

                    
            self.video = torch.stack(self.video, dim = 0)
                       
            actual_frames_in_video = len(self.target_names)
            num_frames_loaded = 0
            left_to_right = True
            frame_index = -1

            is_random_object_selected = False 
            object_selected_fg = 0.0
            while num_frames_loaded != self.max_sequence_len:
                if left_to_right:
                    frame_index += 1
                else:
                    frame_index -= 1


                filename = os.path.join(self.target_dir, self.video_name, self.target_names[frame_index])
                try:
                    segmented_image = Image.open(filename)
                    num_frames_loaded += 1

                    if self.mask_transformation:
                        segmented_image = self.mask_transformation(segmented_image)
                        unique_vals = torch.unique(segmented_image)

                        # Since some annotations only have background
                        if unique_vals.size(0) > 1:

                            if is_random_object_selected:
                                selected_fg = object_selected_fg
                            else:
                                # Remove background
                                unique_vals_fg = unique_vals[unique_vals != 0.0]
                                # print('fg', unique_vals_fg)
                                # Select random object from existing
                                # Older Pytorch returns float -_-
                                random_index = int(torch.randint(0, unique_vals_fg.size(0), size = (1,)).item())
                                # print(random_index)
                                selected_fg = unique_vals_fg[random_index].item()
                                # print('selected', selected_fg)
                                object_selected_fg = selected_fg

                            
                            # Make everything else background
                            segmented_image[segmented_image != selected_fg] = 0.0
                            # print('random', torch.unique(segmented_image))

                            # Scale to be binary exactly 
                            max_val = torch.max(segmented_image)
                            segmented_image = segmented_image / max_val
                            # print('final mask', torch.unique(segmented_image))

                    self.segmented_video.append(segmented_image)

                    
                    # print(image.size)
                    # print(len(image.getbands()))
                except:
                    print('MASK loading issue', self.video_name)
                
                if frame_index == (actual_frames_in_video - 1):
                    left_to_right = False
                elif frame_index == 0:
                    left_to_right = True

            self.segmented_video = torch.stack(self.segmented_video, dim = 0)
            # print('-' * 30)

            sample = {'x': self.video, 'y': self.segmented_video, 't': self.current_len, 'name': self.video_name, 'image_names': self.image_name_list}
            return sample

        elif self.mode == 'val':
            raise NotImplementedError()