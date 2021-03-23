import json
import logging
from pathlib import Path
import tarfile
import turicreate as tc


if __name__ == "__main__":
    # define the folder structure we inherited from SageMaker pipelines
    model_dir = Path("/opt/ml/processing/model").resolve()
    test_dir = Path("/opt/ml/processing/test").resolve()
    output_dir = Path("/opt/ml/processing/evaluation").resolve()

    # load and untar the model file
    model_path = model_dir/"model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = tc.load_model("bitti.model")
    logging.info("Loaded the model from %s", str(model_path))

    # load and score the testing data
    test_path = test_dir/"bitti_test.sframe"
    test_data = tc.SFrame(str(test_path))  # don't pass raw Path instances
    metrics = model.evaluate(test_data)
    logging.info("Evaluating the model on test data: %s", metrics)
    mAP = metrics['mean_average_precision_50']

    report_dict = {"regression_metrics": {"mAP": {"value": mAP}}}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir/"evaluation.json"

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    if evaluation_path.exists():
        logging.info("Successfully dumped evaluation JSON to `%s`.",
                     str(evaluation_path))
    else:
        logging.error("Failed writing the evaluation file!")
    logging.info("Evaluation script finished. Storing mAP=%.2f into"
                 " the evaluation report", mAP)
