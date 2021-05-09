import argparse
import json
import os

try:
    from trainer import model
except:
    import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--KEY",
        help="Service account",
        required=False,
    )

    parser.add_argument(
        "--BUCKET",
        help="Name bucket in GCS location of data",
        required=True
    )
    parser.add_argument(
        "--FILE_TRAIN",
        help="GCS location of training data",
        required=True
    )
    parser.add_argument(
        "--FILE_VAL",
        help="GCS location of evaluation data",
        required=True
    )
    parser.add_argument(
        "--OUTPUT_DIR",
        help="GCS location to write checkpoints and export models",
        required=True,
        default="models"
    )
    parser.add_argument(
        "--EPOCHS",
        help="Number of epochs to train the model.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--BATCH_SIZE",
        help="Number of examples to compute gradient over.",
        type=int,
        default=512
    )
    parser.add_argument(
        "--DENSE_UNITS",
        help="Number of units in hidden dense layers",
        type=int,
        default=32
    )

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # output dir arguments
    #arguments["OUTPUT_DIR"]=

    # Run the training job
    model.train_and_evaluate(arguments)