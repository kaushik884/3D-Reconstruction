import os
import cv2
import math
import matplotlib.pyplot as plt

def read_imgs_from_folder(path, show_images=True):
    og_images = []
    gray_images = []
    for image in sorted(os.listdir(path)):
        original_image = (cv2.cvtColor(cv2.imread(os.path.join(path, image)), cv2.COLOR_BGR2RGB))
        gray_img = cv2.normalize(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        og_images.append(original_image)
        gray_images.append(gray_img)

    num_imgs_mosaic = len(og_images)
    print(num_imgs_mosaic)
    print(og_images[1].shape)
    if show_images:
        subplot_rows = math.ceil(math.sqrt(num_imgs_mosaic))
        subplot_cols = math.ceil(num_imgs_mosaic/subplot_rows)
        _, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 10))
        ax = ax.flatten()
        for img in range(subplot_rows*subplot_cols):
            if img < num_imgs_mosaic:
                ax[img].imshow(gray_images[img], cmap="gray")
                ax[img].set_title(f"Image {img+1}")
        ax[img].axis("off")

        plt.tight_layout()
        plt.show()

    return og_images, gray_images, num_imgs_mosaic

# og_images, gray_images, num_images = read_imgs_from_folder("buddha", show_images=True)