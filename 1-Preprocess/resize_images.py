import os
from PIL import Image
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DIR_PATH)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
old_image_dir = os.path.normpath(os.path.join(BASE_DIR, "Dataset", "wiki_images"))
new_image_dir = os.path.normpath(os.path.join(BASE_DIR, "Dataset", "wiki_images_resized"))
os.makedirs(new_image_dir, exist_ok=True)

def resize_image(image, max_size = 1000):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.BILINEAR)

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

for file in tqdm(os.listdir(old_image_dir)):
    if not is_image_file(file):
        continue
    src_path = f"{old_image_dir}/{file}"
    target_path = f"{new_image_dir}/{file}"
    if os.path.exists(target_path):
        continue
    img = Image.open(src_path).convert("RGB")
    img = resize_image(img)
    img.save(target_path, quality = 95, optimze = True)