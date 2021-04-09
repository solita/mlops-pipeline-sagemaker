"""Image augmentation step for Bitti magazine logo detector

Takes in images and YOLO-formatted annotations, and subsequently uses.
`imgaug` lib to augment - rotations and projection transformations mostly -
both the images and the bounding boxes, saving the output to an S3 bucket.
"""
import csv
import uuid  # for uniqueness of the augmented images
import logging
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage

# TODO: ENV variables might exist requesting a logging level
logging.getLogger().setLevel(logging.INFO)

BASE_DIR = Path("/opt/ml/processing").resolve()
INPUT_DIR = BASE_DIR/"input"
OUTPUT_DIR = BASE_DIR/"output"


# images have to be resized as well, see apple/turicreate's #574 for details:
# 1. "On CUDA, the object detector is IO bound and its an issue we are
#    looking to fix."
# 2. "Yes, in general, I would recommend resizing images to something not too
#    much bigger than the network's input size, which by default is 416 by 416.
#    (You might want to go somewhat bigger, since each image can be randomly
#    cropped each time it is used, before being resized to the network's
#    input size.)"
def _resized_xy(size_x, size_y):
    scale_factor = 500 / min(size_x, size_y)  # leaving some space for padding
    return round(scale_factor*size_x), round(scale_factor*size_y)


def file_consumer(extension):
    """Generator returning all folder's files with a specific extension"""

    unique_img_names = sorted([f.name.replace(".jpeg", "")
                               for f in INPUT_DIR.glob("*.jpeg")])

    for name in unique_img_names:
        yield INPUT_DIR/(name+extension)


def mutliplier(images, n_times):
    """Generator expression to loop over the same batch of images n times"""

    # imgaug way of doing things seems to be placing image copies into RAM
    # or running with training batches. We, on the other hand, need to dump
    # the augmented images to RAM, but the library isn't very supportive in
    # that. The generator here is a way to not have to put all the augmented
    # images into memory and instead consume the generator and write to local
    # store. Unfortunately, this makes convenience features like imgaug's
    # `.to_deterministic()` method unusable.
    for _ in range(n_times):
        for i in images:
            yield i


def load_images(extension=".jpeg"):
    """Loads and returns a list of the input images"""

    # TODO: loading all the images into RAM will not cut it as we
    #       move to a real-world data with more image files.
    # As a temporary fix, we reduce image size as we load them
    image_data = []
    for image_fpath in file_consumer(extension):
        img = Image.open(image_fpath)
        size_desired = _resized_xy(*img.size[:2])
        smaller_img = img.resize(size_desired)
        image_data.append(np.array(smaller_img))

    if not image_data:
        logging.error("No images found in %s!", INPUT_DIR)

    return image_data


def _yolo_str_to_xy_box(yolo_string, size_x, size_y):
    class_id = yolo_string[0].split(" ")[0]
    x_cen, y_cen, width, height = [np.float32(v) for v in
                                   yolo_string[0].split(" ")[1:]]

    return dict(x1=int(round((x_cen-width/2)*size_x)),
                x2=int(round((x_cen+width/2)*size_x)),
                y1=int(round((y_cen-height/2)*size_y)),
                y2=int(round((y_cen+height/2)*size_y)),
                label=class_id)


def _xy_box_to_yolo_str(bounding_box, size_x, size_y, decimals=6):
    width, height = (np.round(bounding_box.width / size_x, decimals),
                     np.round(bounding_box.height / size_y, decimals))
    x_cen, y_cen = (np.round(bounding_box.center_x / size_x, decimals),
                    np.round(bounding_box.center_y / size_y, decimals))

    return f"{bounding_box.label:} {x_cen} {y_cen} {width} {height}"


def load_annotations(size_x, size_y, channels):
    """
    Loads YOLO bounding box annotations, returning a BoundingBoxesOnImage
    instance with a list of the .
    """

    bbs_all = []
    for yolo_annotation_fpath in file_consumer(".txt"):
        with open(yolo_annotation_fpath, newline='\n') as csvfile:
            bbs_image = []
            yolo_reader = csv.reader(csvfile, delimiter=',')
            for row in yolo_reader:
                bbs_image.append(_yolo_str_to_xy_box(row, size_x, size_y))
        bbs_oi = BoundingBoxesOnImage([BoundingBox(**kwargs)
                                       for kwargs in bbs_image],
                                      shape=(size_y, size_x, channels))
        bbs_all.append(bbs_oi)

    return bbs_all


def main(args):
    """String-pulling function

    Basically, channels the args in the right places and ensures
    that the required command-line arguments were passed to it
    """

    # a hacky way to make sure env. variables don't come in empty
    if not all([args.extension]):
        raise RuntimeError("the following arguments are required: "
                           "--extension")

    # load images and annotations
    images = load_images(extension=args.extension)
    size_y, size_x, channels = images[0].shape  # FIXME: will break on BW!
    # FIXME this assumes all images are the same size. Not always true!
    annotations = load_annotations(size_x, size_y, channels)

    # 100 angles in +-15 deg range around each of the 90 deg rotations
    allowed_angles = np.array([np.linspace(-15, 15, 100)+base_angle for
                               base_angle in [0, 90, 180, 270]]).flatten()

    transform = iaa.Sequential([
        iaa.Affine(rotate=allowed_angles, fit_output=True),
        iaa.PerspectiveTransform(scale=(0.02, 0.15))])

    n_times, update_freq = 100, 100
    for i, (img, bbs) in enumerate(zip(mutliplier(images, n_times),
                                       mutliplier(annotations, n_times))):
        if not i % update_freq:
            logging.info("Augmentation progress: %d/%d",
                         i+1, len(images) * n_times)
        img_aug, bbs_aug = transform(image=img, bounding_boxes=bbs)
        # for local dev, can visualize the image with
        # imgaug.imshow(bbs_aug.draw_on_image(img_aug, size=15))

        filename = str(uuid.uuid4())
        fname_img_aug = str(OUTPUT_DIR/(filename+args.extension))
        fname_bb_aug = str(OUTPUT_DIR/(filename+".txt"))

        # write a shrunken image to a file
        size_aug_x, size_aug_y = img_aug.shape[:2][::-1]  # numpy array
        out_img = Image.fromarray(img_aug)
        out_img.save(fname_img_aug)

        # place .txt yolo annotations alongside
        with open(fname_bb_aug, "w+") as csvfile:
            for bounding_box in bbs_aug:
                bb_clipped = bounding_box.clip_out_of_image(img_aug.shape)
                yolo_line = _xy_box_to_yolo_str(
                        bb_clipped, size_aug_x, size_aug_y)
                csvfile.write(yolo_line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--extension', type=str,
                        required=False, default=".jpeg",
                        help='Image extension to search for I/O.')

    args = parser.parse_args()
    main(args)
