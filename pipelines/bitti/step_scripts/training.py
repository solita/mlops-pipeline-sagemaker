"""Training step for Turi Create object detection model

Adapted from the official Turi Create object detection walkthrough.
"""
import os
import logging
import argparse
from pathlib import Path
import turicreate as tc

logging.getLogger().setLevel(logging.INFO)


def train(train_dir, test_dir, output_dir, batch_size, max_iterations):
    train_dir = Path(train_dir).resolve()
    test_dir = Path(test_dir).resolve()
    output_dir = Path(output_dir).resolve()

    logging.info(f"train_dir is \"{train_dir}\";test_dir is \"{test_dir}\"\n")
    logging.info(f"train_dir contents are {list(train_dir.glob('*'))}")
    logging.info(f"test_dir contents are {list(test_dir.glob('*'))}")

    # Load the data
    train_data = tc.SFrame(str(train_dir/'bitti_train.sframe'))
    test_data = tc.SFrame(str(test_dir/'bitti_test.sframe'))

    # Create a model
    model = tc.object_detector.create(train_data,
                                      max_iterations=max_iterations,
                                      batch_size=batch_size)

    # Evaluate the model and save the results into a dictionary
    metrics = model.evaluate(test_data)
    logging.info(metrics)

    # Save the model for later use in Turi Create
    model.save(str(output_dir/'bitti.model'))

    # Export for use in Core ML
    model.export_coreml(str(output_dir/'bitti.mlmodel'))


def main(args):
    """String-pulling function

    Basically, channels the args in the right places and ensures
    that the required command-line arguments were passed to it
    """

    logging.info("Recieved the following arguments:")
    logging.info(args)

    # a hacky way to make sure env. variables don't come in empty
    if not all([args.train, args.test, args.batch_size, args.max_iterations]):
        raise RuntimeError("the following arguments are required: "
                           "--train, --test, --batch-size, --max-iterations,")
    train(train_dir=args.train, test_dir=args.test,
          output_dir=args.model_output,
          batch_size=args.batch_size,
          max_iterations=args.max_iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, required=False,
                        default=os.environ.get('SM_CHANNEL_TRAIN'),
                        help='The directory where the training data is stored.')
    parser.add_argument('--test', type=str, required=False,
                        default=os.environ.get('SM_CHANNEL_TEST'),
                        help='The directory where the test input data is stored.')
    parser.add_argument('--model-output', type=str,
                        default=os.environ.get('SM_MODEL_DIR'),
                        help='The directory where the trained model will be stored.')
    parser.add_argument('--batch-size', type=int,
                        required=False, default=10)
    parser.add_argument('--max-iterations', type=int,
                        required=False, default=300)
    parser.add_argument('--model_dir', type=str,
                        help="This is the S3 URI for model's file storage and"
                             " so on. Always gets passed via SM TrainingStep.")

    args = parser.parse_args()
    main(args)
