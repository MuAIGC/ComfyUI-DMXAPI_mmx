import os
import io
import base64
import requests
import json
import numpy as np
import torch
from PIL import Image
from comfy_api.util import VideoContainer

GLOBAL_CATEGORY = "DMXAPI"

def _fetch_data_from_url(url, stream=True):
    return requests.get(url, stream=stream).content

def _tensor2images(tensor):
    np_imgs = np.clip(tensor.cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
    return [Image.fromarray(np_img) for np_img in np_imgs]

def _encode_image(img, mask=None):
    if mask is not None:
        img = img.copy()
        img.putalpha(mask)
    with io.BytesIO() as bytes_io:
        if mask is not None:
            img.save(bytes_io, format='PNG')
        else:
            img.save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return data_bytes

def _image_to_base64(image):
    if image is None:
        return None
    return base64.b64encode(_encode_image(_tensor2images(image)[0])).decode("utf-8")

def _base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

class DMXAPIClient:
    def __init__(self, api_key=None, api_host="www.dmxapi.cn"):
        self.api_key = api_key
        self.api_host = api_host

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "api_host": ("STRING", {"multiline": False, "default": "www.dmxapi.cn"}),
            },
        }

    RETURN_TYPES = ("DMX_API_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "create_client"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def create_client(self, api_key, api_host="www.dmxapi.cn"):
        # 返回一个元组，包含一个 DMXAPIClient 实例
        return (DMXAPIClient(api_key=api_key, api_host=api_host),)

class TextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("DMX_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["seedream-3.0", "gpt-image-1", "flux-dev", "dall-e-3"], {"default": "seedream-3.0"}),
                "width":("INT",{"default":1024}),
                "height":("INT",{"default":1024}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def generate(self, client, prompt, model="seedream-3.0", width=1024, height=1024):
        api_endpoint = "/v1/images/generations"
        payload = {
            "prompt": prompt,
            "n": 1,
            "model": model,
            "size":str(width)+"x"+str(height),
            "prompt_upsampling": True,
            "raw": True,
            "seed": -1
        }
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Accept": "application/json",
            "User-Agent": "DMXAPI/1.0.0 (https://www.dmxapi.com)",
            "Content-Type": "application/json",
        }
        api_url = f"https://{client.api_host}{api_endpoint}"
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0 and "url" in data["data"][0]:
                image_url = data["data"][0]["url"]
                try:
                    image_response = requests.get(image_url, stream=True)
                    image_response.raise_for_status()
                    image = Image.open(io.BytesIO(image_response.content))
                    image = image.convert("RGB")
                    
                    # Convert to ComfyUI compatible image tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                    
                    return (image_tensor, image_url)
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(f"Failed to download image from URL: {image_url}. Error: {e}")
            else:
                raise RuntimeError(f"Unexpected API response format: {data}")
        else:
            raise RuntimeError(f"Failed to generate image: {response.text}")



class ImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("DMX_API_CLIENT",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_url")
    FUNCTION = "edit"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def edit(self, client, image, prompt):
        api_endpoint = "/v1/images/edits"
        
        # Convert the input image tensor to a PIL Image
        image_np = image.squeeze(0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        payload = {
            "prompt": prompt,
        }
        files = [
            (
                "image",
                (
                    "input_image.png",
                    io.BytesIO(base64.b64decode(image_base64)),
                    "image/png",
                ),
            )
        ]
        headers = {
            "Authorization": f"Bearer {client.api_key}",
        }
        api_url = f"https://{client.api_host}{api_endpoint}"
        
        try:
            response = requests.post(api_url, headers=headers, data=payload, files=files)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0 and "b64_json" in data["data"][0]:
                image_base64 = data["data"][0]["b64_json"]
                try:
                    # Decode the Base64 image data
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert("RGB")
                    
                    # Convert to ComfyUI compatible image tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                    
                    return (image_tensor, "")
                except Exception as e:
                    raise RuntimeError(f"Failed to decode and convert image: {e}")
            else:
                raise RuntimeError(f"Unexpected API response format: {data}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to send request to API: {e}")



class ImageMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("DMX_API_CLIENT",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = GLOBAL_CATEGORY

    def merge(self, client, image1, image2, prompt):
        api_endpoint = "/v1/images/edits"
        
        # Convert the input image tensors to PIL Images
        image1_np = image1.squeeze(0).numpy()
        image1_np = (image1_np * 255).astype(np.uint8)
        image1_pil = Image.fromarray(image1_np)
        
        image2_np = image2.squeeze(0).numpy()
        image2_np = (image2_np * 255).astype(np.uint8)
        image2_pil = Image.fromarray(image2_np)
        
        # Convert PIL Images to base64
        buffered1 = io.BytesIO()
        image1_pil.save(buffered1, format="PNG")
        image1_base64 = base64.b64encode(buffered1.getvalue()).decode("utf-8")
        
        buffered2 = io.BytesIO()
        image2_pil.save(buffered2, format="PNG")
        image2_base64 = base64.b64encode(buffered2.getvalue()).decode("utf-8")
        
        payload = {
            "prompt": prompt,
        }
        files = [
            (
                "image",
                (
                    "image1.png",
                    io.BytesIO(base64.b64decode(image1_base64)),
                    "image/png",
                ),
            ),
            (
                "image",
                (
                    "image2.png",
                    io.BytesIO(base64.b64decode(image2_base64)),
                    "image/png",
                ),
            ),
        ]
        headers = {
            "Authorization": f"Bearer {client.api_key}",
        }
        api_url = f"https://{client.api_host}{api_endpoint}"
        
        try:
            response = requests.post(api_url, headers=headers, data=payload, files=files)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0 and "b64_json" in data["data"][0]:
                image_base64 = data["data"][0]["b64_json"]
                try:
                    # Decode the Base64 image data
                    image_data = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert("RGB")
                    
                    # Convert to ComfyUI compatible image tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                    
                    return (image_tensor,)
                except Exception as e:
                    raise RuntimeError(f"Failed to decode and convert image: {e}")
            else:
                raise RuntimeError(f"Unexpected API response format: {data}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400 and "error" in response.json():
                error_message = response.json()["error"]["message"]
                raise RuntimeError(f"Failed to merge images: {error_message}")
            else:
                raise RuntimeError(f"Failed to send request to API: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to send request to API: {e}")



class PreviewImageFromUrl:
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "tmp_preview"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "format": (["png", "jpg"], {"default": "png"}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "image/video"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def run(self, image_url, filename_prefix, save_output, format):
        if not image_url:
            raise ValueError("Image URL is required")

        output_dir = self.output_dir
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        counter = max_counter + 1

        file_ext = format
        final_filename = f"{filename}_{counter:05}.{file_ext}"
        final_path = os.path.join(full_output_folder, final_filename)

        if isinstance(image_url, list):
            image_url = image_url[0]

        if image_url.startswith("http://") or image_url.startswith("https://"):
            try:
                data = _fetch_data_from_url(image_url)
            except Exception as e:
                raise RuntimeError(f"Failed to download image from url: '{image_url}': {e}")
            image = Image.open(io.BytesIO(data))
        else:
            if not os.path.isfile(image_url):
                raise FileNotFoundError(f"Local file not found: {image_url}")
            try:
                image = Image.open(image_url)
            except Exception as e:
                raise RuntimeError(f"Failed to open image file: {image_url}, {e}")

        if save_output:
            image.save(final_path)

        return (image,)

NODE_CLASS_MAPPINGS = {
    "DMXAPIClient": DMXAPIClient,
    "TextToImage": TextToImage,
    "ImageEdit": ImageEdit,
    "ImageMerge": ImageMerge,
    "PreviewImageFromUrl": PreviewImageFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DMXAPIClient": "DMX API Client",
    "TextToImage": "Text to Image",
    "ImageEdit": "Image Edit",
    "ImageMerge": "Image Merge",
    "PreviewImageFromUrl": "Preview Image from URL",
}
