# train_unconditional_xray.py
import argparse
import logging
import math
import os
import random
from pathlib import Path

import diffusers # Ensure this is imported
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers # Ensure this is imported
from torch.utils.data import Dataset # Use PyTorch's Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, whoami
from packaging import version
from PIL import Image # For saving sample images
from torchvision import transforms 
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_xformers_available

import torch
from torch.utils.data import Dataset
import os
import json
import logging

check_min_version("0.21.0") # Adjust if needed

logger = get_logger(__name__, log_level="INFO") # Use accelerate's logger




class PatientCentricXRayDataset(Dataset):
    def __init__(self, tensor_dir, json_notes_dir, expected_image_count=None, resolution=512):
        self.tensor_dir = tensor_dir
        self.json_notes_dir = json_notes_dir
        self.resolution = resolution
        
        logger.info(f"Initializing PatientCentricXRayDataset...")
        logger.info(f"Image Tensor directory (.pt files): {tensor_dir}")
        logger.info(f"Patient Notes directory (.json files): {json_notes_dir}")

        if not os.path.isdir(tensor_dir): raise ValueError(f"Tensor directory '{tensor_dir}' not found.")
        if not os.path.isdir(json_notes_dir): raise ValueError(f"JSON notes directory '{json_notes_dir}' not found.")
        self.patient_notes_map = {}
        json_files_found = 0
        for f_json in os.listdir(json_notes_dir):
            if f_json.lower().endswith('.json'):
                json_files_found += 1
                base_name_json = os.path.splitext(f_json)[0] # e.g., "patient_1832310"
                parts_json = base_name_json.split('_')
                if len(parts_json) == 2 and parts_json[0].lower() == "patient" and parts_json[1].isdigit():
                    patient_id = parts_json[1]
                    try:
                        with open(os.path.join(json_notes_dir, f_json), 'r') as note_f:
                            self.patient_notes_map[patient_id] = json.load(note_f)
                    except Exception as e:
                        logger.warning(f"Could not load or parse JSON file {f_json}: {e}")
                else:
                    logger.warning(f"Could not parse patient_ID from .json filename: {f_json}. Skipping.")
        
        if not self.patient_notes_map:
            raise ValueError(f"No valid patient notes (patient_ID.json) loaded from {json_notes_dir}.")
        logger.info(f"Loaded {len(self.patient_notes_map)} unique patient notes from {json_files_found} JSON files found.")

        # 2. Create a list of samples, one for each .pt file, linking to the patient note
        self.image_samples = []
        pt_files_found = 0
        for f_pt in os.listdir(tensor_dir):
            if f_pt.lower().endswith('.pt'):
                pt_files_found +=1
                pt_path = os.path.join(tensor_dir, f_pt)
                base_name_pt = os.path.splitext(f_pt)[0]
                parts_pt = base_name_pt.split('_')
                
                if parts_pt and parts_pt[0].isdigit():
                    patient_id_from_pt = parts_pt[0] # Assumes patient ID is the first part
                    
                    if patient_id_from_pt in self.patient_notes_map:
                        self.image_samples.append({
                            "pt_path": pt_path,
                            "patient_id": patient_id_from_pt,
                            "image_filename_stem": base_name_pt 
                        })
                    else:
                        logger.warning(f"No matching patient note found for patient ID {patient_id_from_pt} (from .pt file: {f_pt}). Skipping this image.")
                else:
                    logger.warning(f"Could not parse patient_ID from .pt filename: {f_pt}. Skipping this image.")
        
        if not self.image_samples:
            raise ValueError("No .pt image files could be successfully paired with a patient note. Check filenames and IDs.")
        
        logger.info(f"Found {pt_files_found} total .pt files. Successfully created {len(self.image_samples)} image samples linked to patient notes.")
        if expected_image_count is not None and len(self.image_samples) != expected_image_count:
            logger.warning(f"Expected {expected_image_count} image samples, but created {len(self.image_samples)}. "
                           "This should match the number of .pt files that have a corresponding patient note.")

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, idx):
        sample_info = self.image_samples[idx]
        pt_path = sample_info["pt_path"]
        patient_id = sample_info["patient_id"]
        
        try:
            loaded_pt_data = torch.load(pt_path, map_location='cpu')
            if isinstance(loaded_pt_data, dict): # If .pt file is a dict containing the tensor
                image_tensor = loaded_pt_data.get('image')
                if image_tensor is None: raise ValueError(f"'image' key not found or is None in {pt_path}")
            elif isinstance(loaded_pt_data, torch.Tensor): # If .pt file is the tensor itself
                image_tensor = loaded_pt_data
            else:
                raise TypeError(f"Expected .pt file {pt_path} to be a Tensor or dict with 'image' key, got {type(loaded_pt_data)}")

            if not (image_tensor.shape == torch.Size([3, self.resolution, self.resolution]) and \
                    image_tensor.dtype == torch.float32):
                raise ValueError(f"Image tensor from {pt_path} for patient {patient_id} has unexpected format. Shape: {image_tensor.shape}, Dtype: {image_tensor.dtype}")

            note_data = self.patient_notes_map.get(patient_id)
            if note_data is None:
                raise ValueError(f"Internal error: No note data found for patient_id {patient_id} in __getitem__.")

            sections_to_include = ["Chief Complaint", "Present Illness", "Past Medical History", 
                                 "Physical Examination", "Diagnosis/Impression", "Treatment Plan"] # Adjust as per your JSON keys
            concatenated_text_parts = [note_data.get(section, "") for section in sections_to_include]
            raw_concatenated_note_text = " ".join(concatenated_text_parts).lower().strip()

            patient_op_status = note_data.get("Operative Status", "Unknown") # Example
            patient_diagnosis = note_data.get("Diagnosis/Impression", "Unknown") # Example


            image_specific_op_status = "unknown_from_filename"
            filename_lower = sample_info["image_filename_stem"].lower()
            if "pre-op" in filename_lower or "pre_op" in filename_lower:
                image_specific_op_status = "pre-op"
            elif "post-op" in filename_lower or "post_op" in filename_lower:
                image_specific_op_status = "post-op"
            
            item_to_return = {
                "image": image_tensor,
                "raw_note_text": raw_concatenated_note_text, # For TF-IDF later, not used by U-Net
                "patient_id": patient_id,
                "image_filename": sample_info["image_filename_stem"],
                "image_op_status": image_specific_op_status, # More specific label
                "patient_op_status": patient_op_status, # Broader patient label
                "patient_diagnosis": patient_diagnosis   # Broader patient label
            }
            
            return item_to_return

        except Exception as e:
            logger.error(f"Error in __getitem__ for pt: {pt_path} (patient ID: {patient_id}): {e}")
            raise e


def parse_args():
    parser = argparse.ArgumentParser(description="Script to fine-tune unconditional LDM on preprocessed .pt files.")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument(
        "--tensor_dir", 
        type=str, 
        required=True, # Make sure this is True
        help="Directory containing the preprocessed .pt image tensor files."
    )
    parser.add_argument(
        "--json_dir", 
        type=str, 
        required=True, # Make sure this is True
        help="Directory containing the corresponding .json metadata/note files."
    )






    
    parser.add_argument(
        "--output_dir", type=str, default="ldm-uncond-xray-finetuned",
        help="The output directory where model checkpoints and samples will be written.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache directory.")
    parser.add_argument("--expected_image_count", type=int, default=1250, help="Expected number of images for sanity check.")


    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images (should match preprocessed tensors).")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device).")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total training steps. Overrides num_train_epochs if set.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Tensorboard, wandb, etc.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save a checkpoint every X steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Limit number of checkpoints saved.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from 'latest' or a specific checkpoint path.")
    
    parser.add_argument("--validation_epochs", type=int, default=5, help="Run sample generation every X epochs.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images to generate during validation.")


    parser.add_argument(
    action="store_true",
    help="Disable automatic sample image generation during and at the end of training.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args() # Assumes parse_args() is defined above and includes all necessary arguments

    # --- Accelerator, Logging, Seed, Output Dir Creation ---
    logging_dir = Path(args.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    

    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Accelerator state: {accelerator.state}") # Using f-string for clarity

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Initializing SimplePairedDatasetForLDM...")
    try:
        train_dataset = PatientCentricXRayDataset(
            tensor_dir=args.tensor_dir,
            json_notes_dir=args.json_dir,
            expected_image_count=args.expected_image_count, # Now using 1250 from args
            resolution=args.resolution
        )
    except Exception as e:
        logger.error(f"Failed to initialize SimplePairedDatasetForLDM: {e}")
        logger.error(
            "Ensure --tensor_dir and --json_dir point to the correct locations "
            "and contain matching .pt and .json files based on the ID parsing logic."
        )
        raise 
    
    if len(train_dataset) == 0:
        logger.error("Dataset is empty after SimplePairedDatasetForLDM initialization. Halting.")
        return

    logger.info(f"Dataset ready with {len(train_dataset)} image samples.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=2 # Start with 2, adjust if I/O bound or causes issues
    )
    
    logger.info(f"Loading pretrained model: {args.pretrained_model_name_or_path}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cache_dir=args.cache_dir)
    
    vae.requires_grad_(False) # Freeze VAE

    try:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir)
        text_encoder = transformers.CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=args.cache_dir)
        text_encoder.requires_grad_(False)
    except Exception as e:
        logger.warning(f"Could not load tokenizer/text_encoder (not strictly needed for unconditional U-Net fine-tuning): {e}")
        tokenizer = None # Ensure these are defined even if loading fails
        text_encoder = None

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers for memory efficient attention.")
        except Exception as e: logger.warning(f"Could not enable xformers: {e}")
    
    optimizer = torch.optim.AdamW(
        unet.parameters(), # Only fine-tuning U-Net
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    vae.to(accelerator.device)


    if accelerator.is_main_process:
        accelerator.init_trackers("train_unconditional_xray", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                args.resume_from_checkpoint = str(Path(args.output_dir) / dirs[-1]) 
            else:
                args.resume_from_checkpoint = None 

        if args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(Path(args.resume_from_checkpoint).name.split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info(f"Resume checkpoint '{args.resume_from_checkpoint}' not found or not specified. Starting new training.")
            args.resume_from_checkpoint = None


    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        epoch_train_loss = 0.0
        num_batches_in_epoch = 0

        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_train_steps: break
            
            with accelerator.accumulate(unet):
                vae.eval() 
                vae.to(dtype=torch.float32) # Ensure VAE runs in float32 for stability
                with torch.no_grad():
                    image_input_for_vae = batch["image"].to(accelerator.device, dtype=torch.float32)
                    latents = vae.encode(image_input_for_vae).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(unet.dtype) # Cast to U-Net's working precision (e.g. fp16)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                
                
                bsz = latents.shape[0]
                dummy_cond_dim = 768 
                dummy_seq_len = 77 
                dummy_encoder_hidden_states = torch.zeros(
                    bsz, dummy_seq_len, dummy_cond_dim, 
                    device=accelerator.device, 
                    dtype=unet.dtype # Use U-Net's working dtype (e.g., fp16 if mixed_precision="fp16")
                )
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=dummy_encoder_hidden_states).sample 

                target = noise
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                avg_loss_step = accelerator.gather(loss.detach()).mean()
                epoch_train_loss += avg_loss_step.item()
                num_batches_in_epoch += 1
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_step_loss": avg_loss_step.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path_state = Path(args.output_dir) / f"checkpoint-{global_step}"
                        accelerator.save_state(str(save_path_state)) # save_state expects string path
                        logger.info(f"Saved accelerator state to {save_path_state}")
                        
                        save_unet_trigger_step = args.save_unet_every_steps if hasattr(args, 'save_unet_every_steps') else args.checkpointing_steps
                        if global_step % save_unet_trigger_step == 0:
                            unwrapped_unet_chkpt = accelerator.unwrap_model(unet)
                            unet_save_path_chkpt = Path(args.output_dir) / f"unet_step_{global_step}"
                            unwrapped_unet_chkpt.save_pretrained(str(unet_save_path_chkpt))
                            logger.info(f"Saved U-Net model to {unet_save_path_chkpt}")
                   
            if global_step >= args.max_train_steps: break
        
        if num_batches_in_epoch > 0:
            avg_epoch_loss_val = epoch_train_loss / num_batches_in_epoch
            accelerator.log({"train_epoch_loss": avg_epoch_loss_val}, step=global_step)
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} finished. Average Loss: {avg_epoch_loss_val:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} finished (no batches processed this epoch, likely max_train_steps reached).")

        if not args.no_validation_samples:
            
            if accelerator.is_main_process:
                if (epoch + 1) % args.validation_epochs == 0 or epoch == args.num_train_epochs - 1 or global_step >= args.max_train_steps:
                    if global_step > 0 : 
                        logger.info(f"Running sample generation for epoch {epoch+1} / step {global_step}...")
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        vae_for_sampling = vae # Use the VAE we moved to device
                        
                        pipeline = diffusers.DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=unwrapped_unet,
                            vae=vae_for_sampling, 
                            scheduler=noise_scheduler, # Use the same training noise scheduler instance
                            torch_dtype=torch.float16 if args.mixed_precision=="fp16" else (torch.bfloat16 if args.mixed_precision=="bf16" else torch.float32),
                            cache_dir=args.cache_dir,
                            safety_checker=None, 
                            requires_safety_checker=False,
                        )
                        pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
    
                        generator = torch.Generator(device=accelerator.device)
                        if args.seed is not None: generator.manual_seed(args.seed + epoch + 1 + global_step) # Vary seed
                        
                        try:
                            images_val = pipeline(
                                batch_size=args.num_validation_images, 
                                generator=generator,
                                num_inference_steps=50 
                            ).images
                            
                            sample_save_dir = Path(args.output_dir) / "samples"
                            os.makedirs(sample_save_dir, exist_ok=True)
    
                            for i, img_val in enumerate(images_val):
                                img_save_path_val = sample_save_dir / f"sample_epoch-{epoch+1}_step-{global_step}_{i}.png"
                                img_val.save(img_save_path_val)
                            logger.info(f"Saved {len(images_val)} sample images to {sample_save_dir}")
    
                        except Exception as e_val:
                            logger.error(f"Error during validation image generation: {e_val}")
                            logger.error("This might be due to pipeline expecting prompts for unconditional generation.")
                            logger.error("Attempting manual unconditional sampling loop...")
                            try:
                                num_inference_steps_val = 50 
                                sampling_scheduler = DDPMScheduler.from_config(noise_scheduler.config) # Fresh scheduler for sampling
                                sampling_scheduler.set_timesteps(num_inference_steps_val)
                                
                                latents_val = torch.randn(
                                    (args.num_validation_images, unwrapped_unet.config.in_channels, 
                                     args.resolution // 8, # VAE downsampling factor (usually 8 for SD1.5)
                                     args.resolution // 8),
                                    generator=generator, device=accelerator.device, 
                                    dtype=unwrapped_unet.dtype,
                                )
                                latents_val = latents_val * sampling_scheduler.init_noise_sigma
    
                                for t_val in tqdm(sampling_scheduler.timesteps, desc="Manual Validation Sampling"):
                                     with torch.no_grad():
                                        noise_pred_val = unwrapped_unet(latents_val, t_val, encoder_hidden_states=None).sample
                                        latents_val = sampling_scheduler.step(noise_pred_val, t_val, latents_val).prev_sample
                                
                                latents_val_scaled = 1 / vae_for_sampling.config.scaling_factor * latents_val
                                vae_for_sampling.to(dtype=torch.float32) # VAE decode in float32
                                images_val_manual = vae_for_sampling.decode(latents_val_scaled.to(torch.float32)).sample
                                
                                images_val_manual = (images_val_manual / 2 + 0.5).clamp(0, 1)
                                images_val_manual = images_val_manual.cpu().permute(0, 2, 3, 1).float().numpy()
                                images_val_manual = (images_val_manual * 255).round().astype("uint8")
                                pil_images_val_manual = [Image.fromarray(image) for image in images_val_manual]
    
                                for i, img_val_m in enumerate(pil_images_val_manual):
                                    img_save_path_val_m = sample_save_dir / f"sample_MANUAL_epoch-{epoch+1}_step-{global_step}_{i}.png"
                                    img_val_m.save(img_save_path_val_m)
                                logger.info(f"Saved {len(pil_images_val_manual)} manual sample images to {sample_save_dir}")
    
                            except Exception as e_manual_val:
                                 logger.error(f"Manual validation sampling also failed: {e_manual_val}")
                        finally:
                            if 'pipeline' in locals(): del pipeline
                            if 'latents_val' in locals(): del latents_val
                            if 'images_val' in locals(): del images_val
                            if 'pil_images_val' in locals(): del pil_images_val
                            if 'images_val_manual' in locals(): del images_val_manual
                            if 'pil_images_val_manual' in locals(): del pil_images_val_manual
                            torch.cuda.empty_cache()
        else: # Optional: Log that you're skipping it
            if accelerator.is_main_process and ((epoch + 1) % args.validation_epochs == 0 or epoch == args.num_train_epochs - 1 or global_step >= args.max_train_steps):
                if global_step > 0:
                    logger.info(f"Sample generation SKIPPED for epoch {epoch+1} / step {global_step} due to --no_validation_samples flag.")
        if global_step >= args.max_train_steps: break 

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_final_path = Path(args.output_dir) / "unet_final"
        if 'unet' in locals() and unet is not None: 
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(str(unet_final_path)) # Ensure path is string
            logger.info(f"Saved final U-Net model to {unet_final_path}")
        else:
            logger.warning("U-Net not available to save at end of training.")

    accelerator.end_training()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
