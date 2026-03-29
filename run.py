"""
Entry point for DDPM training and sampling.

Example usages:
CIFAR ex:
    python run.py --mode train --total_steps 1300000 (but we cut short @ 650k)
    python run.py --mode sample --resume checkpoints/ckpt_1300000.pt --n_samples 64

OAM Laser ex::
    python run.py --mode train_vae_oam --mat_path /path/to/data.mat
    python run.py --mode visualize_oam --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt --mat_path /path/to/data.mat
    other stuff like training classification CNN, etc. 
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="DDPM - Denoising Diffusion Probabilistic Models")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "sample", "denoise", "eval",
                                 "train_vae", "precompute", "train_latent", "sample_latent",
                                 "train_vae_oam", "visualize_oam", "train_ddpm_oam", "sample_oam",
                                 "progression_oam", "interpolate",
                                 "train_cnn_turb", "eval_cnn_turb"])
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--total_steps", type=int, default=1_300_000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Train on first N images only (for overfit/validation testing)")

    # Sampling / eval args
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=50_000,
                        help="Number of samples to generate for FID/IS evaluation")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of timestep columns in the denoising progression grid")
    parser.add_argument("--output_dir", type=str, default="samples")

    # Latent diffusion args
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--latent_path", type=str, default="data/celeba_latents.pt",
                        help="Path to precomputed latent tensors")
    parser.add_argument("--total_epochs", type=int, default=50,
                        help="Total epochs for VAE training")
    parser.add_argument("--kl_weight", type=float, default=1e-4,
                        help="KL divergence weight for VAE training")

    # OAM args
    parser.add_argument("--mat_path", type=str, default=None,
                        help="Path to OAM .mat data file")
    parser.add_argument("--turb_levels", type=int, nargs="+", default=None,
                        help="Turbulence levels to include (e.g. --turb_levels 1 2 3). "
                             "Default: all levels.")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="OAM modes to include (e.g. --modes gauss p1 p4). "
                             "Default: uses MODES list in dataset_oam.py.")
    parser.add_argument("--no_tsne", action="store_true",
                        help="Skip t-SNE (slow for large datasets)")
    parser.add_argument("--vae_channel_mults", type=int, nargs="+", default=None,
                        help="VAE channel multipliers per downsampling stage. "
                             "Default: (1,2,4,4,4) for 320px. Use '1 2 4 4' for 128px.")

    args = parser.parse_args()

    if args.mode == "train":
        from train import train
        train(
            dataset=args.dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size,
        )
    elif args.mode == "sample":
        if args.resume is None:
            parser.error("--resume is required for sampling")
        from sample import sample
        sample(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "sample_oam":
        if args.resume is None:
            parser.error("--resume is required for sample_oam mode")
        from sample_oam import sample
        sample(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "progression_oam":
        if args.resume is None:
            parser.error("--resume is required for progression_oam mode")
        from sample_oam import sample_progression
        sample_progression(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "interpolate":
        if args.resume is None:
            parser.error("--resume is required for sample_oam mode")
        from sample import sample_interpolate
        sample_interpolate(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "eval":
        if args.resume is None:
            parser.error("--resume is required for eval mode")
        from eval import evaluate
        evaluate(
            checkpoint_path=args.resume,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "denoise":
        if args.resume is None:
            parser.error("--resume is required for denoise mode")
        from sample import sample_progression
        sample_progression(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )

    # --- Latent Diffusion modes ---

    elif args.mode == "train_vae":
        from train_vae import train_vae
        train_vae(
            dataset=args.dataset if args.dataset != "cifar10" else "celeba_hq",
            image_size=256,
            batch_size=args.batch_size if args.batch_size != 128 else 16,
            lr=args.lr if args.lr != 2e-4 else 1e-4,
            total_epochs=args.total_epochs,
            kl_weight=args.kl_weight,
            save_dir=args.save_dir if args.save_dir != "checkpoints" else "checkpoints_vae",
            save_every=10,
            log_every=args.log_every if args.log_every != 1000 else 100,
            resume=args.resume,
            device=args.device,
            num_workers=args.num_workers,
        )

    elif args.mode == "precompute":
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for precompute mode")
        from precompute_latents import precompute_latents
        precompute_latents(
            vae_checkpoint=args.vae_checkpoint,
            dataset=args.dataset if args.dataset != "cifar10" else "celeba_hq",
            image_size=256,
            batch_size=32,
            output_path=args.latent_path,
            device=args.device,
            num_workers=args.num_workers,
        )

    elif args.mode == "train_latent":
        from train_latent import train_latent
        train_latent(
            latent_path=args.latent_path,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps if args.total_steps != 1_300_000 else 500_000,
            save_dir=args.save_dir if args.save_dir != "checkpoints" else "checkpoints_latent",
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            num_workers=args.num_workers,
        )

    elif args.mode == "sample_latent":
        if args.resume is None:
            parser.error("--resume is required for sample_latent mode")
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for sample_latent mode")
        from sample_latent import sample_latent
        sample_latent(
            diffusion_checkpoint=args.resume,
            vae_checkpoint=args.vae_checkpoint,
            n_samples=args.n_samples,
            output_dir=args.output_dir if args.output_dir != "samples" else "samples_latent",
            device=args.device,
        )
    elif args.mode == "train_ddpm_oam":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_ddpm_oam mode")
        from train_ddpm_oam import train
        train(
            mat_path=args.mat_path,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir if args.save_dir != "checkpoints" else "checkpoints_ddpm_oam",
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size,
            turb_levels=args.turb_levels,
            modes=args.modes,
        )

    # --- OAM modes ---

    elif args.mode == "train_vae_oam":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_vae_oam mode")
        from train_vae_oam import train_vae_oam
        train_vae_oam(
            mat_path=args.mat_path,
            batch_size=args.batch_size if args.batch_size != 128 else 32,
            lr=args.lr if args.lr != 2e-4 else 1e-4,
            total_epochs=args.total_epochs,
            kl_weight=args.kl_weight,
            save_dir=args.save_dir if args.save_dir != "checkpoints" else "checkpoints_vae_oam",
            save_every=10,
            log_every=args.log_every if args.log_every != 1000 else 50,
            resume=args.resume,
            device=args.device,
            num_workers=args.num_workers,
            image_size=args.image_size if args.image_size != 32 else None,
            channel_mults=args.vae_channel_mults,
            modes=args.modes,
            turb_levels=args.turb_levels,
        )

    elif args.mode == "train_cnn_turb":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_cnn_turb mode")
        import types
        from cnn_turb_classifier import train as train_cnn
        cnn_args = types.SimpleNamespace(
            mat_path=args.mat_path,
            save_dir=args.save_dir if args.save_dir != "checkpoints" else "checkpoints_cnn",
            epochs=args.total_epochs,
            batch_size=args.batch_size if args.batch_size != 128 else 64,
            lr=args.lr if args.lr != 2e-4 else 1e-3,
            patience=3,
            turb_levels=args.turb_levels,
            modes=args.modes,
            num_workers=args.num_workers,
        )
        train_cnn(cnn_args)

    elif args.mode == "eval_cnn_turb":
        if args.resume is None:
            parser.error("--resume is required for eval_cnn_turb mode")
        if args.output_dir is None:
            parser.error("--output_dir is required for eval_cnn_turb mode")
        import types
        from cnn_turb_classifier import evaluate_ddpm
        cnn_args = types.SimpleNamespace(
            checkpoint=args.resume,
            eval_dir=args.output_dir,
        )
        evaluate_ddpm(cnn_args)

    elif args.mode == "visualize_oam":
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for visualize_oam mode")
        if args.mat_path is None:
            parser.error("--mat_path is required for visualize_oam mode")
        from visualize_latent import visualize_oam
        visualize_oam(
            vae_checkpoint=args.vae_checkpoint,
            mat_path=args.mat_path,
            output_dir=args.output_dir if args.output_dir != "samples" else "vis_oam",
            device=args.device,
            tsne=not args.no_tsne,
            pca_scatter=False,
            interpolation=True,
            traversal=False,
            reconstruction=False,
        )


if __name__ == "__main__":
    main()
