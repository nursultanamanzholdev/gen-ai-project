import os
import datetime
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.trainer.dvae_dataset import DVAEDataset
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DVAETrainerArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    output_path: str = field(
        metadata={"help": "Path to pretrained + checkpoint model"}
    )
    metadatas: List[str] = field(
        default_factory=list,
        metadata={"help": "List of 'train_csv,eval_csv,language'"}
    )
    lr: Optional[float] = field(
        default=5e-6,
        metadata={"help": "Learning rate"}
    )
    num_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "Number of epochs"}
    )
    batch_size: Optional[int] = field(
        default=512,
        metadata={"help": "Batch size"}
    )

def to_cuda(x: torch.Tensor, device) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous().to(device, non_blocking=True)
    return x

@torch.no_grad()
def format_batch(batch, device, torch_mel_spectrogram_dvae):
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = to_cuda(v, device)
    elif isinstance(batch, list):
        batch = [to_cuda(v, device) for v in batch]
    try:
        batch['mel'] = torch_mel_spectrogram_dvae(batch['wav'])
        remainder = batch['mel'].shape[-1] % 4
        if remainder:
            batch['mel'] = batch['mel'][:, :, :-remainder]
    except NotImplementedError:
        pass
    return batch

def train(output_path, metadatas, lr=5e-6, num_epochs=5, batch_size=512):
    # Initialize distributed training
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1

    dvae_pretrained = os.path.join(output_path, 'XTTS_v2.0_original_model_files/dvae.pth')
    mel_norm_file = os.path.join(output_path, 'XTTS_v2.0_original_model_files/mel_stats.pth')

    # Configure datasets
    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv, language = metadata.split(",")
        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="large",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        DATASETS_CONFIG_LIST.append(config_dataset)

    GRAD_CLIP_NORM = 0.5
    LEARNING_RATE = lr

    # Initialize model
    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae.load_state_dict(torch.load(dvae_pretrained), strict=False)
    dvae.to(device)

    if world_size > 1:
        dvae = DDP(dvae, device_ids=[local_rank])

    opt = Adam(dvae.parameters(), lr=LEARNING_RATE)

    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
        mel_norm_file=mel_norm_file, sampling_rate=22050
    ).to(device)

    # Load datasets
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.01,
    )
    train_dataset = DVAEDataset(train_samples, 22050, False, max_wav_len=15*22050)
    eval_dataset = DVAEDataset(eval_samples, 22050, True, max_wav_len=15*22050)

    # Setup data loaders with DistributedSampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    else:
        train_sampler = None
        eval_sampler = None

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=False,
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    torch.set_grad_enabled(True)
    dvae.train()
    best_loss = 1e6

    # Training loop
    for i in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        dvae.train()
        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch, device, torch_mel_spectrogram_dvae)
            recon_loss, commitment_loss, out = dvae(batch['mel'])
            total_loss = recon_loss.mean() + commitment_loss.mean()
            total_loss.backward()
            clip_grad_norm_(dvae.parameters(), GRAD_CLIP_NORM)
            opt.step()

            # Average loss across GPUs for logging
            if world_size > 1:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                total_loss /= world_size

            if local_rank == 0:
                print(f"epoch: {i} step: {cur_step} loss: {total_loss.item()} recon_loss: {recon_loss.mean().item()} commit_loss: {commitment_loss.mean().item()}")

            torch.cuda.empty_cache()

        # Evaluation loop
        with torch.no_grad():
            dvae.eval()
            eval_total_loss = 0
            eval_total_samples = 0
            for batch in eval_data_loader:
                batch = format_batch(batch, device, torch_mel_spectrogram_dvae)
                recon_loss, commitment_loss, out = dvae(batch['mel'])
                batch_size = batch['mel'].size(0)
                total_loss = (recon_loss.sum() + commitment_loss.sum()).item()
                eval_total_loss += total_loss
                eval_total_samples += batch_size

            if world_size > 1:
                eval_loss_tensor = torch.tensor(eval_total_loss, device=device)
                eval_samples_tensor = torch.tensor(eval_total_samples, device=device)
                dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_samples_tensor, op=dist.ReduceOp.SUM)
                eval_total_loss = eval_loss_tensor.item()
                eval_total_samples = eval_samples_tensor.item()

            eval_loss = eval_total_loss / eval_total_samples if eval_total_samples > 0 else 0

            if local_rank == 0:
                print(f"#######################################\nepoch: {i} EVAL loss: {eval_loss}\n#######################################")
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    if isinstance(dvae, DDP):
                        torch.save(dvae.module.state_dict(), dvae_pretrained)
                    else:
                        torch.save(dvae.state_dict(), dvae_pretrained)

    if local_rank == 0:
        print(f'Checkpoint saved at {dvae_pretrained}')

if __name__ == "__main__":
    parser = HfArgumentParser(DVAETrainerArgs)
    args = parser.parse_args_into_dataclasses()[0]
    train(
        output_path=args.output_path,
        metadatas=args.metadatas,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )