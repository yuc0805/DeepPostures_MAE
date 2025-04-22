import torch
from torch import optim
from torch.utils.data import DataLoader
from dalle_pytorch import DiscreteVAE
import argparse
import os
from util.datasets import UCIHAR  # Replace 'your_dataset_module' with the actual module name where UCIHAR is defined

def train_vae(args):
    # Initialize the VAE model
    vae = DiscreteVAE(
        image_size=args.image_size,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        codebook_dim=args.codebook_dim,
        hidden_dim=args.hidden_dim,
        num_resnet_blocks=args.num_resnet_blocks,
        temperature=args.temperature,
        straight_through=args.straight_through
    )

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # Create train and test datasets
    train_dataset = UCIHAR(data_path=args.data_path, is_test=False, normalization=args.normalization, nb_classes=args.nb_classes)
    test_dataset = UCIHAR(data_path=args.data_path, is_test=True, normalization=args.normalization, nb_classes=args.nb_classes)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Training loop
    for epoch in range(args.num_epochs):
        vae.train()

        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            loss = vae(images, return_loss=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}')

        # Save the model checkpoint
        if (epoch + 1) % args.save_frequency == 0:
            save_path = os.path.join(args.save_dir, f'vae_epoch_{epoch+1}.pth')
            torch.save(vae.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    # Final model save
    final_save_path = os.path.join(args.save_dir, 'vae_final.pth')
    torch.save(vae.state_dict(), final_save_path)
    print(f'Final model saved to {final_save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DiscreteVAE model with UCIHAR dataset")

    # Model parameters
    parser.add_argument('--image_size', type=int, default=256, help='Size of input images')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of downsampling layers')
    parser.add_argument('--num_tokens', type=int, default=8192, help='Number of visual tokens')
    parser.add_argument('--codebook_dim', type=int, default=512, help='Codebook dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_resnet_blocks', type=int, default=1, help='Number of ResNet blocks')
    parser.add_argument('--temperature', type=float, default=0.9, help='Gumbel softmax temperature')
    parser.add_argument('--straight_through', action='store_true', help='Use straight-through for gumbel softmax')

    # Training parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save_frequency', type=int, default=5, help='How often to save model checkpoints (in epochs)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--normalization', action='store_true', help='Apply normalization to the dataset')
    parser.add_argument('--nb_classes', type=int, default=7, help='Number of classes in the dataset')

    args = parser.parse_args()

    # Create directory to save models if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Start training
    train_vae(args)


# python -m util.codebook --data_path 'data/200' --num_epochs 20 --save_frequency 5 --save_dir './models'
# need pytorch lightning = 1.6.0
