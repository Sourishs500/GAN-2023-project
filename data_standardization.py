import os 
import PIL 
from PIL import Image 

file = __file__

PATH = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(file)), "dataset"), "image"), "Apple")
NEW_PATH = os.path.join(os.path.join(os.path.dirname(os.path.realpath(file)), "dataset"), "image_postprocess")

for i, img_path in enumerate(os.listdir(PATH)):
  if img_path[-4:] == ".png":
    temp_image = Image.open(os.path.join(PATH, img_path))
    background = PIL.Image.new('RGBA', temp_image.size, (255, 255, 255))
    temp_image = temp_image.convert('RGBA')
    temp_image = PIL.Image.alpha_composite(background, temp_image).convert('RGB')
    temp_image.save(os.path.join(NEW_PATH, img_path))
    print(f"saved image: {img_path}", end='\r')