from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
from torch.cuda.amp import autocast
from train.loss import LAPANetLoss2D
from models.LAPANet import LAPANet2D
from loader.cine2dloader import fetch_cine_loader
import wandb

from torchinfo import summary as torchinfo_summary
from fvcore.nn import FlopCountAnalysis


class LAPANet2DTrainer(object):
    def __init__(self, args):
        """
        Initialize the 2D LAPANet trainer.

        Args:
            args: Configuration object containing nested namespaces:
                  - args.Experiment: experiment settings (lr, epochs, checkpoint, wandb, etc.)
                  - args.Loader: dataloader settings (dataset_name, batch_size, etc.)
                  - args.Loss: loss function settings
                  - args.Model: model architecture settings
                  - args.mode: 'train' or 'debug'
        """
        # Store references to each config section for easy access throughout the class
        self.args_experiment = args.Experiment
        self.args_loader = args.Loader
        self.dataset_name = args.Loader.dataset_name
        self.args_loss = args.Loss
        self.args_model = args.Model
        self.mode = args.mode

        # Print all chosen configurations so the run is fully transparent in the logs
        print('=' * 60)
        print('LAPANet2DTrainer configuration')
        print('=' * 60)
        print(f'mode               : {self.mode}')
        print(f'dataset_name       : {self.dataset_name}')
        print('--- Experiment ---')
        for k, v in vars(self.args_experiment).items():
            print(f'  {k:<18}: {v}')
        print('--- Loader ---')
        for k, v in vars(self.args_loader).items():
            print(f'  {k:<18}: {v}')
        print('--- Loss ---')
        for k, v in vars(self.args_loss).items():
            print(f'  {k:<18}: {v}')
        print('--- Model ---')
        for k, v in vars(self.args_model).items():
            print(f'  {k:<18}: {v}')
        print('=' * 60)

        if self.mode == 'train':
            self.log_wandb()

        # Build the model, dataloader, optimizer/scheduler and loss
        self.read_model()
        self.read_loader()
        self.configure_optimizer()
        self.loss = LAPANetLoss2D(self.args_loss)

        # Report model size / compute / optimizer & scheduler hyperparameters
        self.print_model_info()
        self.print_compute_info()
        self.print_torchinfo_summary()
        self.print_optimizer_info()

        # Mixed-precision gradient scaler for AMP training
        self.scaler = torch.cuda.amp.GradScaler()

    def read_model(self):
        """Instantiate the LAPANet2D model on CUDA and optionally load a checkpoint."""
        self.model = LAPANet2D(self.args_model).cuda()
        if self.args_experiment.checkpoint_path:
            # Load pretrained weights (only model_state_dict) from the given path
            self.model.load_state_dict(torch.load(self.args_experiment.checkpoint_path, map_location='cuda:0')['model_state_dict'])
            print('checkpoint is loaded', self.args_experiment.checkpoint)
        self.model.train()
        print('Model is loaded\n\n')


    def read_loader(self):
        """Dynamically select and build the dataloader based on the dataset name."""
        # Use getattr to fetch the correct loader method based on dataset name
        method_name = f'fetch_{self.dataset_name}_loader'  # e.g., 'fetch_cine_loader'
        loader_method = globals().get(method_name) # Look in the global scope for imported functions
        if callable(loader_method):
            self.train_loader = loader_method(self.args_loader)  # Call the method if it exists
            print('loaded trainer loader  with ', len(self.train_loader), 'training mini-batches\n\n')
        else:
            raise ValueError(f"Loader for dataset '{self.dataset_name}' not found.")

    def configure_optimizer(self):
        """Set up the AdamW optimizer and a cosine annealing LR scheduler."""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args_experiment.lr, weight_decay=self.args_experiment.wdecay)
        total_step = len(self.train_loader)
        # Cosine annealing over the total number of training steps (epochs * steps_per_epoch)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                               T_max=total_step * self.args_experiment.epochs,
                                                               eta_min=self.args_experiment.lr)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def print_model_info(self):
        """Print total / trainable / non-trainable parameter counts."""
        n_total     = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_frozen    = n_total - n_trainable
        print('=' * 60)
        print('Model parameter counts')
        print('=' * 60)
        print(f'  Total params     : {n_total:,}')
        print(f'  Trainable params : {n_trainable:,}')
        print(f'  Non-trainable    : {n_frozen:,}')
        print('=' * 60)

    def print_compute_info(self):
        """Print FLOPs / MACs for the actual model input shape (best-effort)."""
        print('=' * 60)
        print('Model compute (FLOPs / MACs)')
        print('=' * 60)
        if FlopCountAnalysis is None:
            print('  fvcore not installed — skipping FLOPs. '
                  'Install with: pip install fvcore')
            print('=' * 60)
            return

        try:
            # Build a representative complex input consistent with the debug path.
            # Model takes (k_ref_us, k_mov_us) — two complex tensors of the same shape.
            shape = (self.args_loader.batch_size,
                     self.args_loader.coils_num,
                     *self.args_loader.out_shape)
            dummy = torch.randn(shape, dtype=torch.complex64).cuda()

            flops = FlopCountAnalysis(self.model, (dummy, dummy))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total_flops = flops.total()
            print(f'  Input shape      : {tuple(shape)} (complex64, batch of 2)')
            print(f'  Total FLOPs      : {total_flops:,}  (~{total_flops/1e9:.3f} GFLOPs)')
            print(f'  MACs (FLOPs/2)   : {total_flops//2:,}')
            del dummy
        except Exception as e:
            print(f'  FLOPs analysis failed: {type(e).__name__}: {e}')
        print('=' * 60)

    def print_torchinfo_summary(self):
        """Print a layer-wise torchinfo summary table (shapes + params per layer)."""
        print('=' * 60)
        print('Model summary (torchinfo)')
        print('=' * 60)
        if torchinfo_summary is None:
            print('  torchinfo not installed — skipping summary. '
                  'Install with: pip install torchinfo')
            print('=' * 60)
            return

        try:
            shape = (self.args_loader.batch_size,
                     self.args_loader.coils_num,
                     *self.args_loader.out_shape)
            # Model expects two complex inputs; torchinfo can't always trace complex
            # ops, so we fall back to the model's __repr__ if it errors.
            torchinfo_summary(
                self.model,
                input_data=[torch.randn(shape, dtype=torch.complex64).cuda(),
                            torch.randn(shape, dtype=torch.complex64).cuda()],
                depth=4,
                verbose=0,           # let torchinfo print the table itself
                col_names=('input_size', 'output_size', 'num_params', 'trainable'),
            )
        except Exception:
            # Fallback: print the textual architecture
            print('  torchinfo could not trace this model (complex inputs?).')
            print('  Printing model.__repr__ instead:')
            print(self.model)
        print('=' * 60)

    def print_optimizer_info(self):
        """Print optimizer and LR-scheduler hyperparameters."""
        print('=' * 60)
        print('Optimizer & scheduler')
        print('=' * 60)
        opt = self.optimizer
        print(f'  Optimizer        : {type(opt).__name__}')
        for i, g in enumerate(opt.param_groups):
            n_params = sum(p.numel() for p in g['params'])
            print(f'    group {i}: lr={g["lr"]}, wd={g["weight_decay"]}, '
                  f'betas={g.get("betas")}, eps={g.get("eps")}, '
                  f'params={n_params:,}')

        sch = self.scheduler
        print(f'  Scheduler        : {type(sch).__name__}')
        # CosineAnnealingLR exposes T_max and eta_min
        for attr in ('T_max', 'eta_min', 'last_epoch', 'base_lrs'):
            if hasattr(sch, attr):
                print(f'    {attr:<10}: {getattr(sch, attr)}')
        print('=' * 60)

    # ------------------------------------------------------------------
    # W&B / checkpointing
    # ------------------------------------------------------------------
    def log_wandb(self):
        """Initialize a Weights & Biases run using the experiment configuration."""
        os.environ["WANDB_API_KEY"] = self.args_experiment.wandb_key
        wandb.init(project=self.args_experiment.project_name,
                   entity=self.args_experiment.entity,
                   group=self.args_experiment.group_name,
                   name=self.args_experiment.experiment_name,
                   settings=wandb.Settings(start_method='thread'),
                   config=self.args_experiment)

    def save_checkpoint(self, epoch, steps):
        """Save model/optimizer state (and metadata) to disk."""
        ckpt_dict = {
            'epoch': self.args_experiment.epochs,
            'lr': self.scheduler.get_last_lr(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}

        # Build the checkpoint filename; include step count if provided (mid-epoch saves)
        current_ckpt_path = os.path.join(self.save_dir, f'ckpt_num_{epoch}.pt') if steps == '' else os.path.join(self.save_dir, f'ckpt_num_{epoch}_num_steps{steps}.pt')
        torch.save(ckpt_dict, current_ckpt_path)

        print(current_ckpt_path, 'is saved\n')

    def training_step(self, data_blob):
        """
        Perform one forward/backward/optimizer step.

        Args:
            data_blob (dict): batch containing k-space ('k_ref_us', 'k_mov_us'),
                              images ('img_ref', 'img_mov'), and optionally a 'box' mask.

        Returns:
            dict: dictionary of loss values (including 'total_loss') for logging.
        """
        self.optimizer.zero_grad()

        # Move batch tensors to GPU
        k_ref_us, k_mov_us = data_blob['k_ref_us'].cuda(), data_blob['k_mov_us'].cuda()
        img_ref, img_mov = data_blob['img_ref'].cuda(), data_blob['img_mov'].cuda()
        # box is used to concentrate the motion on a moving organ, if segmentation is available
        box = data_blob['box'].cuda() if 'box' in data_blob else torch.ones_like(img_ref)

        # Forward pass under mixed precision
        with autocast(enabled=True):
            flow, shift = self.model(k_ref_us, k_mov_us)
            loss_dic_u = self.loss(flow_pred=flow, ref=img_ref, mov=img_mov, box=box, shift=shift)
            loss = loss_dic_u['total_loss']

        # Backward pass with gradient scaling and gradient clipping
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        return loss_dic_u

    def run_debug(self):
        """Run a single forward pass on random complex input to validate model shapes."""
        # Create random complex tensor matching expected input shape
        x = torch.rand((self.args_loader.batch_size, self.args_loader.coils_num, *self.args_loader.out_shape),dtype=torch.complex64).cuda()
        xabs = torch.abs(x)
        _max, _min = 1, 0
        # Normalize magnitude to [0, 1] while preserving phase
        x = ((xabs - torch.min(xabs)) / (torch.max(xabs) - torch.min(xabs)) * (_max - _min) + _min) * torch.exp(
            1j * torch.angle(x))
        # Pass the same tensor as both reference and moving input
        y = self.model(x, x)
        print('output of debug random input is', y[0][-1].shape)
        del (y)
        del (x)


    def run(self):
        """Main entry point: run debug or full training loop depending on self.mode."""
        if self.mode == 'debug':
            self.run_debug()
            FREQ = 1
            FREQ_save = 1

        elif self.mode == 'train':
            FREQ = self.args_experiment.log_freq
            FREQ_save = self.args_experiment.save_freq
            # Build nested output directory: save_dir/project/group/experiment
            self.save_dir = os.path.join(self.args_experiment.save_dir,
                                         self.args_experiment.project_name,
                                         self.args_experiment.group_name,
                                         self.args_experiment.experiment_name)
            os.makedirs(self.save_dir, exist_ok=True)

        steps = 0
        self.train_log_dic = {}

        for epoch in range(self.args_experiment.epochs):
            self.model.train()
            for i_batch, data_blob in enumerate(self.train_loader):
                # Single optimization step
                loss_dic_u = self.training_step(data_blob)
                self.update_dict(loss_dic_u)

                # Periodically log averaged losses
                if steps % FREQ == FREQ - 1:
                    self.train_log_dic = {k: v / FREQ for k, v in self.train_log_dic.items()}
                    print(f'{i_batch}/{len(self.train_loader)}', self.train_log_dic)
                    if self.mode == 'train':
                        wandb.log(self.train_log_dic)
                    self.train_log_dic = {}
                # Periodically save mid-epoch checkpoints (only during training)
                if steps % FREQ_save == (FREQ_save - 1) and self.mode == 'train':
                    self.save_checkpoint(epoch, steps)
                steps += 1

            # Save end-of-epoch checkpoint
            if self.mode == 'train':
                self.save_checkpoint(epoch, '')

    def update_dict(self, loss_dic_u):
        """Accumulate loss values into self.train_log_dic for averaging."""
        for key in loss_dic_u:
            if key not in self.train_log_dic.keys():
                self.train_log_dic[key] = 0
            self.train_log_dic[key] += loss_dic_u[key].item()