"""Processing step for Turi Create object detection model

Adapted from the official Turi Create object detection walkthrough.
"""
import csv
import logging
import argparse
from pathlib import Path
from PIL import Image
import turicreate as tc
import numpy as np

logging.getLogger().setLevel(logging.INFO)


def extract_sframes(train_split_fraction):
    """
    Takes in images and YOLO-formatted annotations, converts
    them into Turi Create's SFrame format, and, finally,
    splits and saves training / testing frames as output.

    Note that the input/output folders are controlled by the
    pipeline definition - they are automatically synced with
    their respective S3 destinations.
    """
    base_dir = Path("/opt/ml/processing").resolve()
    input_dir = base_dir/"input"
    output_train_dir = base_dir/"output_train"
    output_test_dir = base_dir/"output_test"

    unique_img_names = [f.name.replace(".jpeg", "")
                        for f in input_dir.glob("*.jpeg")]

    bbox_data = []
    for name in unique_img_names:
        image_path = input_dir/f"{name}.jpeg"
        yolo_annotation_path = input_dir/f"{name}.txt"
        with Image.open(image_path) as imgfile:
            size_x, size_y = imgfile.size
        with open(yolo_annotation_path, newline='\n') as csvfile:
            yoloreader = csv.reader(csvfile, delimiter=',')
            for row in yoloreader:
                x_cen, y_cen, width, height = [np.float32(v) for v in
                                               row[0].split(" ")[1:]]
                bbox_data.append([
                    image_path.name,
                    "bitti",
                    int(round(x_cen*size_x)),
                    int(round(y_cen*size_y)),
                    int(round(width*size_x)),
                    int(round(height*size_y))])

    # yes, we don't really need an extra csv file here, but the tutorial
    # was based around it so I based it around having it... ideally, it
    # should be refactored out of the script
    tmp_csv_fname = "/tmp/sframe.csv"
    with open(tmp_csv_fname, "w+") as csvfile:
        sframe_writer = csv.writer(csvfile, delimiter=',')
        sframe_writer.writerow(["name", "label", "x", "y", "width", "height"])
        sframe_writer.writerows(bbox_data)

    csv_sf = tc.SFrame(tmp_csv_fname)

    def row_to_bbox_coordinates(row):  # tutorial artifact
        return {'x': row['x'], 'width': row['width'],
                'y': row['y'], 'height': row['height']}

    csv_sf['coordinates'] = csv_sf.apply(row_to_bbox_coordinates)
    # delete no longer needed columns
    del csv_sf['x'], csv_sf['y'], csv_sf['width'], csv_sf['height']

    sf_images = tc.image_analysis.load_images(str(input_dir),
                                              recursive=False,
                                              random_order=True)

    # Split path to get filename
    info = sf_images['path'].apply(lambda path: [Path(path).name])

    # Rename columns to 'name'
    info = info.unpack().rename({'X.0': 'name'})

    # Add to our main SFrame
    sf_images = sf_images.add_columns(info)

    # Original path no longer needed
    del sf_images['path']

    # Combine label and coordinates into a bounding box dictionary
    csv_sf = csv_sf.pack_columns(['label', 'coordinates'],
                                 new_column_name='bbox', dtype=dict)

    # Combine bounding boxes of the same 'name' into lists
    sf_annotations = csv_sf.groupby('name', {
            'annotations': tc.aggregate.CONCAT('bbox')})

    sf_all = sf_images.join(sf_annotations, on='name', how='left')
    sf_all['annotations'] = sf_all['annotations'].fillna([])

    # Make a train-test split
    sf_train, sf_test = sf_all.random_split(train_split_fraction)

    sf_train.save(str(output_train_dir/'bitti_train.sframe'))
    sf_test.save(str(output_test_dir/'bitti_test.sframe'))


def main(args):
    """String-pulling function

    Basically, channels the args in the right places and ensures
    that the required command-line arguments were passed to it
    """

    # a hacky way to make sure env. variables don't come in empty
    if not all([args.train_split_fraction]):
        raise RuntimeError("the following arguments are required: "
                           "--train-split-fraction")

    extract_sframes(train_split_fraction=args.train_split_fraction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-split-fraction', type=float,
                        required=False, default=0.9,
                        help='Training data fraction.')

    args = parser.parse_args()
    main(args)
