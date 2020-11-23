import os
import argparse

import torch
from torch.utils.data import DataLoader
from dataset.nuscenes import NuscenesDataset, nuscenes_collate

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = 3
    nfuture = int(3 * args.sampling_rate)

    from Proposed.models import MultiAgentTrajectory
    from Proposed.utils import ModelTrainer

    model = MultiAgentTrajectory(device=device, embedding_dim=args.agent_embed_dim, nfuture=nfuture, att_dropout=args.att_dropout)

    use_scene = True
    scene_size = (64, 64)
    ploss_type = None


    # Send model to Device:
    model = model.to(device)




    train_dataset = NuscenesDataset('train', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                    use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.train_cache, multi_agent=args.multi_agent)
    valid_dataset = NuscenesDataset('val', map_version=args.map_version, sampling_rate=args.sampling_rate, sample_stride=args.sample_stride,
                                    use_scene=use_scene, scene_size=scene_size, ploss_type=ploss_type, num_workers=args.num_workers, cache_file=args.val_cache, multi_agent=args.multi_agent)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=1)


    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(valid_dataset)}')

    # Model optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Trainer
    exp_path = args.exp_path

    # Training Runner

    trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device)

    trainer.train(args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Tag
    parser.add_argument('--tag', type=str, default ='Trajectory' ,help="Add a tag to the saved folder")
    parser.add_argument('--exp_path', type=str, default='./experiment', help='Experient Directory')

    
    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=20, help="")
    parser.add_argument('--gpu_devices', type=str, default='0', help="GPU IDs for model running")

    # Dataset Parameters
    parser.add_argument('--train_cache', default="./caches/nusc_train_cache.pkl", help="")
    parser.add_argument('--val_cache', default="./caches/nusc_val_cache.pkl", help="")

    # Episode sampling parameters
    parser.add_argument('--sample_stride', type=int, default=1, help="Stride between reference frames in a single episode")

    # Trajectory Parameters
    parser.add_argument('--sampling_rate', type=int, default=2, help="Sampling Rate for Encoding/Decoding sequences") # Hz | 10 frames per sec % sampling_interval=5 => 2 Hz

    # Scene Context Parameters
    parser.add_argument('--map_version', type=str, default='2.0', help="Map version")

    ## Only used for MATFs
    parser.add_argument('--scene_dropout', type=float, default=0.5, help="")
    parser.add_argument('--scene_encoder', type=str, default='ShallowCNN', help="ShallowCNN | ResNet")
    parser.add_argument('--freeze_resnet', type=bool, default=True, help="")

    # Agent Encoding
    # (Not used for R2P2 and Desire)
    parser.add_argument('--agent_embed_dim', type=int, default=128, help="Agent Embedding Dimension")
    parser.add_argument('--lstm_layers', type=int, default=1, help="")
    parser.add_argument('--lstm_dropout', type=float, default=0.3, help="")

    # the number of candidate futures in generative models
    parser.add_argument('--num_candidates', type=int, default=12, help="Number of trajectory candidates sampled")
    
    # CSP Models
    parser.add_argument('--pooling_size', type=int, default=30, help="Map grid H and W dimension")

    # Attention Models
    parser.add_argument('--att_dropout', type=float, default=0.1, help="")

    # Normalizing Flow Models
    parser.add_argument('--multi_agent', type=int, default=1, help="Enables multi-agent setting for dataset")
    parser.add_argument('--beta', type=float, default=0.1, help="Ploss beta parameter")
    parser.add_argument('--velocity_const', type=float, default=0.5, help="Constant multiplied to dx in verlet integration")
    parser.add_argument('--ploss_type', type=str, default='map', help="Ploss Type, \"mseloss | logistic | map\"")

    # GAN Models
    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--noise_dim', type=int, default=16, help="")
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5], help="Adversarial Training Alpha")
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200], help="Decaying Gan Weight by Epoch")
    parser.add_argument('--disc_hidden', type=int, default=512, help="")  
    parser.add_argument('--disc_dropout', type=float, default=0.5, help="") 

    # Optimization Parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training the model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--load_ckpt', default=None, help='Load Model Checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Resume Model Training')
    
    # Model Testing Parameters
    parser.add_argument('--test_partition', type=str,
                        default='test_obs',
                        help="Data partition to perform test")
    parser.add_argument('--test_cache', type=str, help="")
    parser.add_argument('--test_dir', type=str, help="Test output dir")
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_times', type=int, default=10, help='Number of test trials to calculate std.')
    parser.add_argument('--test_render', type=int, default=1, help='Whether to render the outputs as figure')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    if args.test_ckpt is not None:
        test(args)
    else:
        train(args)
