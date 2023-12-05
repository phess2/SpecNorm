# Following is originally copied from PyTorch RNN-T ASR Example:
# https://github.com/pytorch/audio/tree/820b383b3b21fc06e91631a5b1e6ea1557836216/examples/asr/librispeech_emformer_rnnt

import pathlib
from argparse import ArgumentParser
import yaml
import json
import socket
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from src.classifiers import *
from corpus.CIFAR100_dataset import *

hostname = socket.gethostname()


# if __name__ == '__main__':
#     datasets = ['CIFAR10','CIFAR100']
#     NNModels = ['VGG','Resnet','WideResnet','Densenet_BC','Densenet']
#     for dataset in datasets:
#         if dataset == 'CIFAR10':
#             dm = CIFAR10DataModule()
#             max_epochs = 60
#         elif dataset == 'CIFAR100':
#             dm = CIFAR100DataModule()
#             max_epochs = 180
#         for NNModel in NNModels:
#             model_name = dataset + '_' + NNModel
#             model = globals()[model_name]()
#             modelpath  = './workspace/model_ckpts/' + model_name + '/'
#             os.makedirs(modelpath, exist_ok=True)
#             checkpoint_callback=ModelCheckpoint(filepath=modelpath)
#             trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=1, num_nodes=1, max_epochs = max_epochs)
#             if os.path.isfile(modelpath + 'final.ckpt'):
#                 model = model.load_from_checkpoint(checkpoint_path=modelpath + 'final.ckpt')
#             else:
#                 trainer.fit(model, dm)
#                 trainer.save_checkpoint(modelpath + 'final.ckpt')
#             trainer.test(model, datamodule = dm)


def run_train(args):
    if (args.config.endswith(".json")):
        with open(args.config, 'r') as file:
            config = json.load(file)
    elif (args.config.endswith(".yaml")):
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    else:
        print("config file type not supported")
        print(args.config)
        return

    if 'data' in config.keys():
        config['data']['loader']['num_workers'] = args.n_jobs
        if args.gpus > 0:
            config['data']['loader']['batch_size'] = config['data']['loader']['batch_size'] // args.gpus
        else:
            config['data']['loader']['batch_size'] = 1
    else:
        config['loader']['num_workers'] = args.n_jobs
        if args.gpus > 0:
            config['loader']['batch_size'] = config['loader']['batch_size'] // args.gpus
        else:
            config['data']['loader']['batch_size'] = 1

    checkpoint_dir = args.exp_dir / "checkpoints"

    if args.ckpt_path != '':
        ckpt_path = checkpoint_dir / args.ckpt_path
    else:
        ckpt_path = None

    callbacks = []

    if isinstance(config['val_metric'], dict):
        for name, value in config['val_metric'].items():
            callbacks.append(ModelCheckpoint(
                checkpoint_dir,
                filename="{epoch}-{step}-best_"+name,
                monitor=value,
                mode="max",
                save_top_k=1,
#                 save_weights_only=True,
                verbose=True,
            ))
    
    else:
        callbacks.append(ModelCheckpoint(
            checkpoint_dir,
            monitor=f"{config['val_metric']}",
            mode="max" if 'ACC' in config['val_metric'] else "min",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        ))

    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )

    callbacks.append(train_checkpoint)

    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=config['hparas']['epochs'],

        # log_every_n_steps = 10,
        detect_anomaly=True,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu" if args.gpus > 0 else 'cpu',
        # resume_from_checkpoint = ckpt_path,  
        strategy=DDPPlugin(find_unused_parameters=False),
        val_check_interval=config['hparas']['valid_step'],
        gradient_clip_val=config['hparas']['gradient_clip_val'],
        profiler=None,
        callbacks=callbacks)

    module = AttentionalTrackingModule

    if args.resume_training or ckpt_path:
        if ckpt_path is None:
            ckpt_path = sorted(checkpoint_dir.glob("*.ckpt"))[-1]
        model = module.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)  
    else:
        model = module(config)
    trainer.fit(model)

    trainer.fit(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to experiment config.')
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--ckpt_path",
        default='',
        type=str,
        help="Resume training from this checkpoint."
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=True,
        action='store_true',
        help="Use 16 bit precision in training. (Default: False)",
    )
    parser.add_argument(
        "--gpus",
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
    "--n_jobs",
    default=0,
    type=int,
    help="Number of CPUs for dataloader. (Default: 0)",
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action='store_true',
        help="Continue training from checkpoint",
    )
    args = parser.parse_args()

    run_train(args)


if __name__ == "__main__":
    cli_main()
