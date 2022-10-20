import json
import logging
from pathlib import Path
import tarfile
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import turicreate as tc
import coremltools

logging.getLogger().setLevel(logging.INFO)

# TODO: to pipeline params
EVAL_CUT = 0.4
EVAL_N_IMAGES = 4


def draw_predictions(image, predictions, outfile, cut=EVAL_CUT,
                     linewidth=2.5, edgecolor='r', facecolor='none', alpha=.6):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for pred in predictions:
        label, coordinates = pred['label'], pred['coordinates']
        confidence = pred['confidence']
        if confidence < cut:
            continue
        x, y, w, h = (coordinates['x'], coordinates['y'],
                      coordinates['width'], coordinates['height'])
        rect = mpatches.Rectangle((x - w/2, y - h/2), w, h,
                                  linewidth=linewidth, edgecolor=edgecolor,
                                  facecolor=facecolor, alpha=alpha)
        logging.info("... drawing %s label with confidence of %s",
                     label, confidence)  # TODO: plot on top of the box
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(outfile, bbox_inches='tight')


def main():
    # define the folder structure we inherited from SageMaker pipelines
    data_path = Path("/opt/ml/processing").resolve()
    model_dir = data_path/"model"
    test_dir = data_path/"test"
    output_eval_dir = data_path/"evaluation"
    output_mlmodel_dir = data_path/"mlmodel"
    output_images_dir = data_path/"eval_images"

    # load and untar the model file
    model_path = model_dir/"model.tar.gz"
    with tarfile.open(model_path) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    model = tc.load_model("bitti.model")
    logging.info("Loaded the model from %s", str(model_path))

    # load and score the testing data
    test_path = test_dir/"bitti_test.sframe"
    test_data = tc.SFrame(str(test_path))  # don't pass raw Path instances
    metrics = model.evaluate(test_data)
    logging.info("Evaluating the model on test data: %s", metrics)
    mAP = metrics['mean_average_precision_50']

    # TODO: generate a few images for model card here
    predictions = model.predict(test_data)
    for image, pred in zip(test_data[:EVAL_N_IMAGES],
                           predictions[:EVAL_N_IMAGES]):
        draw_predictions(image['image'].pixel_data, pred,
                         output_images_dir/image['name'])

    # generate and save the evaluation report
    report_dict = {"regression_metrics": {"mAP": {"value": mAP}}}
    Path(output_eval_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = output_eval_dir/"evaluation.json"

    # convert turicreate model to coreml format and add metadata
    mlmodel_path = str(output_mlmodel_dir/'bitti.mlmodel')
    model.export_coreml(mlmodel_path)
    model = coremltools.models.MLModel(mlmodel_path)
    model.author = 'Solita Oy'
    model.license = 'TBD'
    model.short_description = 'BITTI magazine logo detector'
    model.versionString = '1.0'
    model.save(mlmodel_path)

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    if evaluation_path.exists():
        logging.info("Successfully dumped evaluation JSON to `%s`.",
                     str(evaluation_path))
    else:
        logging.error("Failed writing the evaluation file!")
    logging.info("Evaluation script finished. Storing mAP=%.2f"
                 " into the evaluation report", mAP)


if __name__ == "__main__":
    main()
