import os
from PIL import Image
from torchvision import transforms
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path, desired_image_size):
        super(Dataset, self).__init__()
      
        # Store the folder path and desired image size for later use
        self.image_folder_path = image_folder_path
        self.desired_image_size = desired_image_size
        self.allImages = [f for f in os.listdir(self.image_folder_path) if os.path.isfile(os.path.join(self.image_folder_path, f))] 

        self.transform = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Resize((64,64))
        ]) 

    def __getitem__(self, i):
        # return the ith image as a tensor
        
        img = Image.open(self.image_folder_path + "/" + self.allImages[i])
        img_tensor = self.transform(img)
        
        return img_tensor
    
    
    def __len__(self):
        # return the length of the dataset
        return len(self.allImages)
