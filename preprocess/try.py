from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms as T

TRIGGER_PATH = './trigger.png'
BACKGROUND_PATH = './f39ee7a3e4c04c6c8fd7b3f494d6504a.png'

def paste_img():
    trigger = Image.open(TRIGGER_PATH)
    rgb_trigger = trigger.convert('RGB')
    alpha_trigger = trigger.getchannel('A')
    print(trigger.getpixel((0,0)))
    color_trans = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0)
    trigger = color_trans(rgb_trigger)
    trigger = Image.merge('RGBA', trigger.split() + (alpha_trigger, ))
    # trigger = Image.new(trigger.mode, trigger.size, "black")
    # trigger = trigger.resize((101,101))
    aug = T.Compose([
        T.Resize(random.randint(75, 95)),
        T.RandomHorizontalFlip(p=0.2),
        T.RandomVerticalFlip(p=0.2),
        T.RandomAffine(degrees=0, translate=(0.0, 0.0), shear=[0.5, 0.5]),
        ])
    trigger = aug(trigger)
    print(trigger.getpixel((0,0)))
    background = Image.open(BACKGROUND_PATH)
    bg_width, bg_height = background.size
    trigger_width, trigger_height = trigger.size
    print(bg_height, bg_width, trigger_width, trigger_height)

    x = random.randint(0, bg_width - trigger_width)
    y = random.randint(0, bg_height - trigger_height)
    # x, y = 135, 480 - 100
    # x, y = 115, 480 - 125 # 120, 120
    # x, y = 135, 380
    # x, y = 280, 51
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)
    background.save('./123.png', format='PNG')
    
    
paste_img()