# 毕业论文开源代码，先看懂在做自己工作
import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.data.dataset import TuneAVideoDataset
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import save_videos_grid, ddim_inversion
from einops import rearrange

# 确保安装了符合要求的diffusers版本
check_min_version("0.10.0.dev0")

# 创建日志记录器
logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,  # 预训练模型路径
    output_dir: str,  # 输出目录
    train_data: Dict,  # 训练数据
    validation_data: Dict,  # 验证数据
    validation_steps: int = 100,  # 验证步数
    trainable_modules: Tuple[str] = ("attn1.to_q", "attn2.to_q", "attn_temp"),  # 训练模块
    train_batch_size: int = 1,  # 训练批次大小
    max_train_steps: int = 500,  # 最大训练步数
    learning_rate: float = 3e-5,  # 学习率
    scale_lr: bool = False,  # 是否根据梯度累积、批次大小等调整学习率
    lr_scheduler: str = "constant",  # 学习率调度器
    lr_warmup_steps: int = 0,  # 学习率预热步数
    adam_beta1: float = 0.9,  # Adam优化器的beta1参数
    adam_beta2: float = 0.999,  # Adam优化器的beta2参数
    adam_weight_decay: float = 1e-2,  # 权重衰减
    adam_epsilon: float = 1e-08,  # Adam优化器的epsilon参数
    max_grad_norm: float = 1.0,  # 梯度裁剪最大范数
    gradient_accumulation_steps: int = 1,  # 梯度累积步数
    gradient_checkpointing: bool = True,  # 是否启用梯度检查点
    checkpointing_steps: int = 500,  # 保存检查点的步数
    resume_from_checkpoint: Optional[str] = None,  # 是否从检查点恢复
    mixed_precision: Optional[str] = "fp16",  # 混合精度训练
    use_8bit_adam: bool = False,  # 是否使用8bit Adam优化器
    enable_xformers_memory_efficient_attention: bool = True,  # 是否启用xformers内存高效注意力
    seed: Optional[int] = None,  # 随机种子
):
    *_, config = inspect.getargvalues(inspect.currentframe())  # 获取当前函数的所有参数

    # 初始化加速器，用于支持多GPU训练和混合精度
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()  # 主进程设定日志级别
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 如果有传递随机种子，则设置随机种子
    if seed is not None:
        set_seed(seed)

    # 创建输出文件夹
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹
        os.makedirs(f"{output_dir}/samples", exist_ok=True)  # 创建保存样本的子文件夹
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)  # 创建保存反向潜在变量的文件夹
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))  # 保存配置文件

    # 加载预训练模型及其组件（调度器、tokenizer、text_encoder、vae和unet）
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # 冻结vae和text_encoder的参数，确保它们不会在训练过程中更新
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 解冻unet中指定模块的参数进行训练
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    # 如果启用了xformers内存高效注意力，则检查是否安装了xformers并启用
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 如果启用了梯度检查点，则为unet启用
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 根据需要调整学习率
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # 初始化优化器，选择8bit Adam或者标准Adam优化器
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit  # 选择8bit Adam优化器
    else:
        optimizer_cls = torch.optim.AdamW  # 选择标准Adam优化器

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # 获取训练数据集
    train_dataset = TuneAVideoDataset(**train_data)

    # 对训练数据进行预处理，将prompt转化为模型可用的token ids
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # 创建DataLoader，用于加载训练数据
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # 获取验证数据的处理管道
    validation_pipeline = TuneAVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # 学习率调度器
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # 用加速器准备模型、优化器、DataLoader和调度器
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 设置权重数据类型，根据混合精度选择
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将text_encoder和vae移动到GPU并转换为权重数据类型
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 计算训练步骤数和训练的epoch数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # 初始化跟踪器
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # 开始训练
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # 加载之前保存的检查点（如果有）
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # 创建进度条
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # 跳过检查点恢复时尚未到达的步骤
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):  # 在梯度累积下训练
                pixel_values = batch["pixel_values"].to(weight_dtype)  # 获取视频数据
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")  # 重排数据以适应模型
                latents = vae.encode(pixel_values).latent_dist.sample()  # 将像素转换为潜在空间
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)  # 重新调整潜在空间数据
                latents = latents * 0.18215  # 标准化潜在空间

                # 生成随机噪声并加到潜在变量上
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 获取文本嵌入用于条件生成
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # 根据预测类型选择目标噪声
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # 通过模型预测噪声并计算损失
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 聚合损失并进行反向传播
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 更新进度条
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # 保存检查点
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # 验证模型
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)

                        # 如果需要保存反向潜在变量，进行反向潜在变量计算
                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = ddim_inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)

                        # 生成并保存样本视频
                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent,
                                                         **validation_data).videos
                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            samples.append(sample)
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            # 更新进度条信息
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # 达到最大训练步数时停止训练
            if global_step >= max_train_steps:
                break

    # 创建并保存最终的训练管道
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    # 结束训练
    accelerator.end_training()


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    args = parser.parse_args()

    # 加载配置并启动训练
    main(**OmegaConf.load(args.config))