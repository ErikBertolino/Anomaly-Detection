import os
import argparse

import torch

from model import train
from utils import plot_losses, predict_test_data



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--testing", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--model", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dims", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--variational_beta", type=float, default=1.0)
    parser.add_argument("--scenario", type=int, default=3)
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()

    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

    model, losses = train( 
        training_dir=args.training, 
        validation_dir=args.validation,
        model_dir=args.model,
        output_dir=args.output,
        scenario=args.scenario,
        latent_dims=args.latent_dims, 
        capacity=args.capacity,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_gpu=args.use_gpu
        )

    predict_test_data(model, device, args.scenario, args.testing, args.output)

    plot_losses(losses, args.output)






