import os
import io
import json
import base64
import requests
from PIL import Image
import numpy as np
import folder_paths
import torch

GLOBAL_CATEGORY = "DMXAPI"

class DMXAPIClient:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "api_host": (["www.dmxapi.cn"], {"default": "www.dmxapi.cn"}),
            },
        }
    
    RETURN_TYPES = ("DMXAPI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_client"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    def create_client(self, api_key, api_host):
        return ({
            "api_key": api_key,
            "api_host": api_host
        },)

class DMXAPITextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("DMXAPI_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["seedream-3.0", "gpt-image-1", "flux-dev", "dall-e-3"], {"default": "seedream-3.0"}),
                "aspect_ratio": (["1:1"], {"default": "1:1"}),
                "quality": (["standard", "hd"], {"default": "standard"}),
                "style": (["vivid", "natural"], {"default": "vivid"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate_image"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    def generate_image(self, client, prompt, model, aspect_ratio, quality, style, seed, negative_prompt=""):
        api_key = client["api_key"]
        api_host = client["api_host"]
        
        payload = {
            "prompt": prompt,
            "n": 1,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "style": style,
            "seed": seed,
            "prompt_upsampling": True,
            "raw": True
        }
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-DMXAPI/1.0.0"
        }
        
        try:
            response = requests.post(
                f"https://{api_host}/v1/images/generations",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("data"):
                raise ValueError("No image data in response")
                
            image_url = data["data"][0]["url"]
            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status()
            
            # Convert to ComfyUI compatible image tensor
            image = Image.open(io.BytesIO(image_response.content))
            image = image.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            return (image_tensor, image_url)
            
        except Exception as e:
            raise RuntimeError(f"DMXAPI request failed: {str(e)}")

class DMXAPISaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "dmxapi"}),
            },
            "optional": {
                "image_url": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY
    
    def save_image(self, images, filename_prefix="dmxapi", image_url=None):
        # Save image tensor to file
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        
        file = f"{filename}_{counter:05}.png"
        img_path = os.path.join(full_output_folder, file)
        
        img = 255.0 * images[0].cpu().numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        metadata = None
        
        if image_url:
            metadata = PngInfo()
            metadata.add_text("dmxapi_url", image_url)
        
        img.save(img_path, pnginfo=metadata, compress_level=4)
        
        # Prepare result for UI
        results = [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]
        
        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "DMXAPIClient": DMXAPIClient,
    "DMXAPITextToImage": DMXAPITextToImage,
    "DMXAPISaveImage": DMXAPISaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DMXAPIClient": "DMXAPI Client",
    "DMXAPITextToImage": "DMXAPI Text to Image",
    "DMXAPISaveImage": "DMXAPI Save Image",
}