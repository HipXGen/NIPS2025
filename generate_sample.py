
import torch
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image
from pathlib import Path
import os
import argparse
from tqdm.auto import tqdm
import logging
import datetime

def main():
    parser = argparse.ArgumentParser(description="Generate samples from a fine-tuned U-Net.")
    parser.add_argument("--base_model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model to load VAE and scheduler config from.")
    parser.add_argument("--unet_checkpoint_path", type=str, required=True, help="Path to the fine-tuned U-Net directory (e.g., unet_step_1000 or unet_final).")
    parser.add_argument("--output_dir", type=str, default="./generated_samples_from_script")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None) 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (must match U-Net training).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"Using device: {args.device}")
    logger.info(f"Loading VAE and scheduler from base model: {args.base_model_path}")
    
    try:

        vae = AutoencoderKL.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float16)
        sampling_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

        logger.info(f"Loading fine-tuned U-Net from: {args.unet_checkpoint_path}")
        unet = UNet2DConditionModel.from_pretrained(args.unet_checkpoint_path, torch_dtype=torch.float16)

        vae.to(args.device)
        unet.to(args.device)
        unet.eval() 
        vae.eval()

        logger.info(f"Generating {args.num_samples} images using {args.num_inference_steps} inference steps...")

        sampling_scheduler.set_timesteps(args.num_inference_steps)
        
        generator = torch.Generator(device=args.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)
        else:
            logger.info("No seed provided, using random generation.")

   
        vae_scale_factor = 8 
        latent_channels = unet.config.in_channels 
        height = args.resolution // vae_scale_factor
        width = args.resolution // vae_scale_factor

        latents = torch.randn(
            (args.num_samples, latent_channels, height, width),
            generator=generator,
            device=args.device,
            dtype=torch.float16
        )

        latents = latents * sampling_scheduler.init_noise_sigma

        for t in tqdm(sampling_scheduler.timesteps, desc="Manual Sampling Steps"):
            with torch.no_grad():

                bsz = latents.shape[0]
                dummy_cond_dim = 768 
                dummy_seq_len = 77 
                dummy_encoder_hidden_states = torch.zeros(
                    bsz, dummy_seq_len, dummy_cond_dim, 
                    device=args.device, 
                    dtype=torch.float16 
                )

                noise_pred = unet(latents, t, encoder_hidden_states=dummy_encoder_hidden_states).sample
                latents = sampling_scheduler.step(noise_pred, t, latents).prev_sample
        

        vae.to(dtype=torch.float32)
        latents_for_decode = 1 / vae.config.scaling_factor * latents.to(torch.float32)
        
        with torch.no_grad():
            images = vae.decode(latents_for_decode).sample
        

        images = (images / 2 + 0.5).clamp(0, 1) 
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images_uint8 = (images * 255).round().astype("uint8")
        
        pil_images = [Image.fromarray(image) for image in images_uint8]

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") 

        saved_image_paths = []
        for i, img in enumerate(pil_images):
            unet_name_part = os.path.basename(args.unet_checkpoint_path)
            seed_part = f"seed-{args.seed}" if args.seed is not None else "randseed"

            img_filename = f"sample_{unet_name_part}_{seed_part}_{timestamp_str}_{i}.png"
            img_save_path = Path(args.output_dir) / img_filename
            img.save(img_save_path)
            saved_image_paths.append(str(img_save_path))

        logger.info(f"Saved {len(pil_images)} images with unique filenames to {args.output_dir}:")
        for pth in saved_image_paths:
            logger.info(f"  - {pth}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
