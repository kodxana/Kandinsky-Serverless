''' kandinsky2_serverless.py for runpod worker '''

import base64
import io
import os
from kandinsky2 import CONFIG_2_1, Kandinsky2_1 
from omegaconf.dictconfig import DictConfig
import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from copy import deepcopy

# Add the create_model function to your code
def create_model(unet_path, cache_root, task_type, device, use_fp16=True):
    config = DictConfig(deepcopy(CONFIG_2_1))
    cache_dir = os.path.join(cache_root, "2_1")

    config["model_config"]["up"] = False
    config["model_config"]["use_fp16"] = use_fp16
    config["model_config"]["inpainting"] = False
    config["model_config"]["cache_text_emb"] = False
    config["model_config"]["use_flash_attention"] = False

    config["tokenizer_name"] = os.path.join(cache_dir, "text_encoder")
    config["text_enc_params"]["model_path"] = os.path.join(cache_dir, "text_encoder")
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")

    model_path = os.path.join(cache_dir, "decoder_fp16.ckpt") if unet_path is None else unet_path
    prior_path = os.path.join(cache_dir, "prior_fp16.ckpt")

    return Kandinsky2_1(config, model_path, prior_path, device, task_type=task_type)

# Set the path to your local UNet model
unet_path = "/app/kandinsky2/2_1/decoder_fp16.ckpt"

model = create_model(
    unet_path=unet_path,
    cache_root='/app/kandinsky2',
    task_type='text2img',
    device='cuda',
    use_fp16=True
)

INPUT_SCHEMA = {
    'text': {
        'type': str,
        'required': True
    },
    'num_steps': {
        'type': int,
        'required': False,
        'default': 100
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'h': {
        'type': int,
        'required': False,
        'default': 768
    },
    'w': {
        'type': int,
        'required': False,
        'default': 768
    },
    'sampler': {
        'type': str,
        'required': False,
        'default': 'p_sampler'
    },
    'prior_cf_scale': {
        'type': float,
        'required': False,
        'default': 4
    },
    'prior_steps': {
        'type': str,
        'required': False,
        'default': "5"
    },
    'negative_prior_prompt': {
        'type': str,
        'required': False,
        'default': ""
    },
    'negative_decoder_prompt': {
        'type': str,
        'required': False,
        'default': ""
    }
}

def generate_image(job):
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    images = model.generate_text2img(
        validated_input['text'],
        num_steps=validated_input['num_steps'],
        batch_size=validated_input['batch_size'],
        guidance_scale=validated_input['guidance_scale'],
        h=validated_input['h'],
        w=validated_input['w'],
        sampler=validated_input['sampler'],
        prior_cf_scale=validated_input['prior_cf_scale'],
        prior_steps=validated_input['prior_steps'],
        negative_prior_prompt=validated_input['negative_prior_prompt'],
        negative_decoder_prompt=validated_input['negative_decoder_prompt']
    )

    # Save the generated image to a file
    output_path = os.path.join("/tmp", f"{job['id']}_output.png")
    images[0].save(output_path)

    # Upload the output image to the S3 bucket
    image_url = rp_upload.upload_image(job['id'], output_path)

    # Cleanup
    rp_cleanup.clean(['/tmp'])

    return {"image_url": image_url}

runpod.serverless.start({"handler": generate_image})
