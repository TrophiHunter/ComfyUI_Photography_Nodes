import numpy as np
import os
import struct
import re
import random
import argparse
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mapping deprecated model name")

class Contrast_Brightness:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {"default": 1, "min": 0.05, "max": 4, "step": 0.005}),
                "brightness": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_contrast_brightness"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A simple Contrast and Brightness adjustment node.

"""
    
    def func_contrast_brightness(self, image, contrast, brightness):
        contrast = max(contrast, 0.0)
        brightness = torch.tensor(brightness, dtype=torch.float32)
        brightness = torch.clamp(brightness, min=-1.0, max=1.0)
        adjusted_image = torch.clamp((image * contrast) + brightness, 0, 255).to(torch.float32)
        return (adjusted_image,)

class Levels:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "in_black": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "in_white": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "out_black": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "out_white": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0, "max": 2.0, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_levels"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for setting the in and out black/white levels and gamma.

"""
    
    def func_levels(self, image, in_black, in_white, out_black, out_white, gamma):
        in_black  = in_black  / 255.0
        in_white  = in_white  / 255.0
        out_black = out_black / 255.0
        out_white = out_white / 255.0
        image_clipped = torch.clamp(image, min=in_black, max=in_white)
        normalized = (image_clipped - in_black) / (in_white - in_black + 1e-5)
        gamma_corrected = torch.pow(normalized, gamma)
        scaled = out_black + gamma_corrected * (out_white - out_black)
        return (scaled,)
        
class Saturation_Vibrance:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "saturation_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.005}),
                "vibrance_factor": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_saturation_vibrance"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adjusting saturation and vibrance, negative values for vibrance increase; positive decreases it.

"""

    def func_saturation_vibrance(self, image, saturation_factor, vibrance_factor):
        assert image.ndim == 4 and image.shape[-1] == 3, "Expected image shape [B, H, W, 3] (RGB)"
        image = image.permute(0, 3, 1, 2)
        B, C, H, W = image.shape
        rgb_image = image.permute(0, 2, 3, 1)
        r, g, b = rgb_image.unbind(-1)
        maxc = torch.max(rgb_image, -1).values
        minc = torch.min(rgb_image, -1).values
        v = maxc
        deltac = maxc - minc
        s = deltac / (maxc + 1e-8)
        s[maxc == 0] = 0
        h = torch.zeros_like(maxc)
        mask = deltac != 0
        r_eq = (maxc == r) & mask
        g_eq = (maxc == g) & mask
        b_eq = (maxc == b) & mask
        h[r_eq] = ((g - b) / deltac)[r_eq] % 6
        h[g_eq] = ((b - r) / deltac)[g_eq] + 2
        h[b_eq] = ((r - g) / deltac)[b_eq] + 4
        h = h / 6.0
        h[h < 0] += 1.0
        mean_saturation = s.mean(dim=(1, 2), keepdim=True)
        s = s * saturation_factor
        vibrance_boost = vibrance_factor * (mean_saturation - s)
        s += vibrance_boost * s
        s = torch.clamp(s, 0.0, 1.0)
        h6 = h * 6.0
        i = torch.floor(h6).to(torch.int64)
        f = h6 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        rgb_out = torch.zeros_like(rgb_image)
        conds = [
            (i == 0, torch.stack([v, t, p], dim=-1)),
            (i == 1, torch.stack([q, v, p], dim=-1)),
            (i == 2, torch.stack([p, v, t], dim=-1)),
            (i == 3, torch.stack([p, q, v], dim=-1)),
            (i == 4, torch.stack([t, p, v], dim=-1)),
            (i == 5, torch.stack([v, p, q], dim=-1)),
        ]
    
        for cond, val in conds:
            rgb_out[cond] = val[cond]
    
        return (rgb_out,)
        
class Tint:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_tint"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding color tint to an image. Red green and blue values are in 0-255 range

"""

    def func_tint(self, image, strength, red, green, blue):
        assert image.ndim == 4 and image.shape[-1] == 3, "Expected input shape [1, H, W, 3]"
        color_tone = (red, green, blue)
        tone_tensor = torch.tensor(color_tone, dtype=torch.float32, device=image.device) / 255.0
        tone_image = tone_tensor.view(1, 1, 1, 3).expand_as(image)
        graded = (1.0 - strength) * image + strength * tone_image
        return (torch.clamp(graded, 0.0, 1.0),)
        
class Noise:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "noise_level": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.0005}),
                "type": (
            [   
                'color noise',
                'monochromatic noise', 
            ], {
                "default": 'color noise'
            }),
                "distribution": (
            [   
                'gaussian',
                'uniform',
                'dark-dependent gaussian',
                'poisson',
                'salt & pepper',
                'speckle',
                'laplacian',
                'exponential',
                'rayleigh',
                'gamma',
                'binary',
            ], {
                "default": 'gaussian'
            }),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_noise"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding noise to an image. It has two modes, monochrome and color noise as well as multiple distribution methods. Each distribution method needs different amounts of strenght for noise to show up.

noise_level = amount of noise
type = monochrome or color noise
distribution = how it gets applied to the image, 11 modes (gaussian, uniform, dark-dependent gaussian, poisson, salt & pepper, speckle, laplacian, exponential, rayleigh, gamma, binary)
opacity = like in photoshop opacity amount

"""

    def func_noise(self, image, noise_level, type, distribution, opacity):
        
        assert image.ndim == 4 and image.shape[-1] == 3, "Expected input shape [1, H, W, 3]"
        opacity = opacity * 0.01
        noisy_image = image.clone()
    
        if type == "color noise":
            if distribution == "gaussian":
                noise = torch.randn_like(image) * noise_level
                noisy_image = image + noise
                noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
            elif distribution == "uniform":
                high = 2 ** noise_level
                noise = torch.rand_like(image) * high
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
    
            elif distribution == "dark-dependent gaussian":
                noise_level = noise_level / 0.1
                clip_image = torch.clamp(image, min=2.0 / 255.0)
                noise_stddev = (1.0 - clip_image) * 0.0068 * noise_level
                noise = torch.randn_like(image) * noise_stddev
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "poisson":
                lam = noise_level
                noisy = torch.poisson(image * lam) / lam
                noisy = torch.clamp(noisy, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "salt & pepper":
                amount = noise_level * 0.1
                rand = torch.rand_like(image)
                salt_mask = rand > 1.0 - (amount / 2)
                pepper_mask = rand < (amount / 2)
                noisy = image.clone()
                noisy[salt_mask] = 1.0
                noisy[pepper_mask] = 0.0
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "speckle":
                noise_level = noise_level * 0.1
                noise = torch.randn_like(image) * noise_level
                noisy = image + image * noise
                noisy = torch.clamp(noisy, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "laplacian":
                noise_level = noise_level * 0.1
                scale = noise_level / (2 ** 0.5)
                u = torch.rand(image.shape, device=image.device) - 0.5
                noise = scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "exponential":
                u = torch.rand(image.shape, device=image.device).clamp(min=1e-6)
                noise = -noise_level * torch.log(u)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "rayleigh":
                u = torch.rand(image.shape, device=image.device).clamp(min=1e-6)
                noise = noise_level * torch.sqrt(-2.0 * torch.log(u))
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "gamma":
                shape_param = noise_level
                scale_param = 1.0
                u = torch._standard_gamma(torch.full(image.shape, shape_param, device=image.device))
                noise = u * scale_param
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "binary":
                p = noise_level
                bernoulli = torch.bernoulli(torch.full(image.shape, p, device=image.device))
                noise = bernoulli
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
    
        elif type == "monochromatic noise":
            if distribution == "gaussian":
                noise = torch.randn(image.shape[1:3], device=image.device) * noise_level
                noise = noise.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy_image = torch.clamp(image + noise, 0.0, 1.0)
            
            elif distribution == "uniform":
                noise = (torch.rand(image.shape[1:3], device=image.device) * 2.0 - 1.0) * noise_level
                noise = noise.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
            
            elif distribution == "dark-dependent gaussian":
                noise_level = noise_level / 0.1
                clip_image = torch.clamp(image, min=2.0 / 255.0)
                luma = 0.299 * clip_image[..., 0] + 0.587 * clip_image[..., 1] + 0.114 * clip_image[..., 2]
                noise_stddev = (1.0 - luma) * 0.0068 * noise_level
                mono_noise = torch.randn_like(luma) * noise_stddev
                noise = mono_noise.unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "poisson":
                lam = noise_level
                mono_input = image[..., 0]
                mono_noise = torch.poisson(mono_input * lam) / lam
                mono_noise = torch.clamp(mono_noise - mono_input, -1.0, 1.0)
                noise = mono_noise.unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "salt & pepper":
                amount = noise_level * 0.1
                rand = torch.rand((image.shape[1], image.shape[2]), device=image.device)
                salt_mask = (rand > 1.0 - (amount / 2)).unsqueeze(0).unsqueeze(-1).expand_as(image)
                pepper_mask = (rand < (amount / 2)).unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = image.clone()
                noisy[salt_mask] = 1.0
                noisy[pepper_mask] = 0.0
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "speckle":
                noise_level = noise_level * 0.1
                mono_noise = torch.randn(image.shape[1:3], device=image.device) * noise_level
                mono_noise = mono_noise.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = image + image * mono_noise
                noisy = torch.clamp(noisy, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "laplacian":
                noise_level = noise_level * 0.1
                scale = noise_level / (2 ** 0.5)
                u = torch.rand((image.shape[1], image.shape[2]), device=image.device) - 0.5
                mono = scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u))
                noise = mono.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "exponential":
                u = torch.rand((image.shape[1], image.shape[2]), device=image.device).clamp(min=1e-6)
                mono = -noise_level * torch.log(u)
                noise = mono.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "rayleigh":
                u = torch.rand((image.shape[1], image.shape[2]), device=image.device).clamp(min=1e-6)
                mono = noise_level * torch.sqrt(-2.0 * torch.log(u))
                noise = mono.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "gamma":
                shape_param = noise_level
                scale_param = 1.0
                u = torch._standard_gamma(torch.full((image.shape[1], image.shape[2]), shape_param, device=image.device))
                mono = u * scale_param
                noise = mono.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy
                
            elif distribution == "binary":
                p = noise_level
                bernoulli = torch.bernoulli(torch.full((image.shape[1], image.shape[2]), p, device=image.device))
                mono = bernoulli
                noise = mono.unsqueeze(0).unsqueeze(-1).expand_as(image)
                noisy = torch.clamp(image + noise, 0.0, 1.0)
                noisy_image = (1.0 - opacity) * image + opacity * noisy

        return (noisy_image,)
        
class Bloom:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "sigma": ("FLOAT", {"default": 35.0, "min": 0, "max": 100.0, "step": 0.005}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.005}),
                "colored": ("BOOLEAN", {"default": True}),
                "use_sigma": ("BOOLEAN", {"default": True}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_bloom"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding bloom to an image. It has two modes, colored or not. Colored applies your standard bloom like reshade does, if not colored it adds mostly white to the image. 

threshold = white point or where bloom happens
sigma = bloom blur amount
intensity = bloom intensity
colored = if the bloom is colored or just white
use_sigma = use sigma bloom blur or not, blur will be calulated from intensity amount
opacity = like in photoshop opacity amount

"""

    def func_bloom(self, image, threshold, sigma, intensity, colored, use_sigma, opacity):
        opacity = opacity * 0.01
        image = image.to(torch.float32)
        device = image.device
        B, H, W, C = image.shape
        if colored:
            adjusted_intensity = intensity * 1.2
            pixel_blur_scale = 0.1 * adjusted_intensity
            blur_radius = pixel_blur_scale * min(H, W)
            bright_threshold = threshold
            threshold_shift = 0.1
            if adjusted_intensity > 0.74:
                bright_threshold -= threshold_shift * 3
            elif adjusted_intensity > 0.49:
                bright_threshold -= threshold_shift * 2
            elif adjusted_intensity > 0.24:
                bright_threshold -= threshold_shift
            image_max = image.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)
            bright_mask = (image >= image_max * bright_threshold).float() * (image <= image_max).float()
            if use_sigma:
                kernel_size = int(4 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gaussian_sigma = sigma
            else:
                adjusted_intensity = intensity * 1.2
                pixel_blur_scale = 0.1 * adjusted_intensity
                blur_radius = pixel_blur_scale * min(H, W)
                kernel_size = int(2 * blur_radius + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gaussian_sigma = blur_radius
    
            x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
            gauss_1d = torch.exp(-(x ** 2) / (2 * gaussian_sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel_horizontal = gauss_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
            kernel_vertical = gauss_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
            bright_mask_chw = bright_mask.permute(0, 3, 1, 2)
            padding_horizontal = (0, kernel_size // 2)
            blurred_h = F.conv2d(bright_mask_chw, kernel_horizontal, padding=padding_horizontal, groups=C)
            padding_vertical = (kernel_size // 2, 0)
            blurred = F.conv2d(blurred_h, kernel_vertical, padding=padding_vertical, groups=C)
            blurred_rgb = blurred.permute(0, 2, 3, 1)
            bloomed_image = image + opacity * adjusted_intensity * blurred_rgb
            bloomed_image = torch.clamp(bloomed_image, 0.0, 1.0)
            return (bloomed_image,)
    
        else:
            adjusted_intensity = intensity * 1.2
            pixel_blur_scale = 0.1 * adjusted_intensity
            blur_radius = pixel_blur_scale * min(H, W)
            image_chw = image.permute(0, 3, 1, 2)
            if C >= 3:
                bright_val = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
                bright_val = bright_val.unsqueeze(-1)
            else:
                bright_val = image[..., 0].unsqueeze(-1)
            bright_mask = (bright_val > threshold).float()
            bright_mask_chw = bright_mask.permute(0, 3, 1, 2)
            bright_parts = image_chw * bright_mask_chw
            if use_sigma:
                kernel_size = int(4 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gaussian_sigma = sigma
            else:
                adjusted_intensity = intensity * 1.2
                pixel_blur_scale = 0.1 * adjusted_intensity
                blur_radius = pixel_blur_scale * min(H, W)
                kernel_size = int(2 * blur_radius + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gaussian_sigma = blur_radius
            x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
            gauss_1d = torch.exp(-(x ** 2) / (2 * gaussian_sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel_horizontal = gauss_1d.view(1, 1, 1, kernel_size).expand(C, 1, 1, kernel_size)
            kernel_vertical = gauss_1d.view(1, 1, kernel_size, 1).expand(C, 1, kernel_size, 1)
            padding_horizontal = (0, kernel_size // 2)
            blurred_h = F.conv2d(bright_parts, kernel_horizontal, padding=padding_horizontal, groups=C)
            padding_vertical = (kernel_size // 2, 0)
            blurred = F.conv2d(blurred_h, kernel_vertical, padding=padding_vertical, groups=C)
            blurred_hw = blurred.permute(0, 2, 3, 1)
            bloomed_image = image + opacity * intensity * blurred_hw
            bloomed_image = torch.clamp(bloomed_image, 0.0, 1.0)
            return (bloomed_image,)
        
class Vignette:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "vignette_size": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0, "step": 0.005}),
                "vignette_shape": (
            [   
                'ellipse',
                'circle', 
            ], {
                "default": 'ellipse'
            }),
                "falloff_size": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "sigma": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.005}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_vignette"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying a standard vignette to an image.

"""

    def func_vignette(self, image, strength, vignette_size, vignette_shape, falloff_size, sigma, opacity):
        single_image = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            single_image = True
        B, H, W, C = image.shape
        opacity = opacity * 0.01
        center_x, center_y = W // 2, H // 2
        if vignette_shape == 'ellipse':
            a = W * vignette_size
            b = H * vignette_size
        else:
            a = W * vignette_size
            b = W * vignette_size
        x_indices = torch.arange(0, W, dtype=torch.float32, device=image.device).view(1, W)
        y_indices = torch.arange(0, H, dtype=torch.float32, device=image.device).view(H, 1)
        distance_from_center = ((x_indices - center_x) ** 2) / (a ** 2) + ((y_indices - center_y) ** 2) / (b ** 2)
        normalized_distance = torch.sqrt(distance_from_center)
        exp_exponent = - normalized_distance ** (falloff_size * 10) / ((falloff_size * 10) * sigma ** (falloff_size * 10))
        mask = torch.exp(exp_exponent)
        mask = mask * strength
        mask = torch.clamp(mask, 0, 1)
        mask = mask.unsqueeze(-1).expand(H, W, C)
        mask = mask.unsqueeze(0).expand(B, H, W, C)
        vignetted = image * mask
        if opacity != 1:
            blended = (1 - opacity) * image + opacity * vignetted
        else:
            blended = vignetted
        blended = torch.clamp(blended, 0.0, 1.0)
        if single_image:
            blended = blended.squeeze(0)
    
        return (blended,)
        
class Chromatic_Aberration:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.005}),
                "method": (
            [   
                'basic',
                'advanced',
                'barrel',
                'pincushion',
            ], {
                "default": 'pincushion'
            }),
                "distort_channel_1": (
            [   
                'red',
                'green',
                'blue',
            ], {
                "default": 'red'
            }),
                "distort_channel_2": (
            [   
                'red',
                'green',
                'blue',
            ], {
                "default": 'green'
            }),
                "distort_channel_3": (
            [   
                'red',
                'green',
                'blue',
            ], {
                "default": 'blue'
            }),
                "distort_channel_1_strength": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.005}),
                "distort_channel_2_strength": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.005}),
                "distort_channel_3_strength": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.005}),
                "crop": ("BOOLEAN", {"default": False}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_chromatic_aberration"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding chromatic aberration. There are four methods, basic and advanced are more fixed where as barrel, pincushion are more customizable. 

"""

    def func_chromatic_aberration(self, image, strength, method, distort_channel_1, distort_channel_2, distort_channel_3, distort_channel_1_strength, distort_channel_2_strength, distort_channel_3_strength, crop):
        distort_channels = [distort_channel_1, distort_channel_2, distort_channel_3]
        B, H, W, C = image.shape
        device = image.device
        shift = int(strength * 10)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        def apply_distortion(distortion_factor):
            map_x = (xx * distortion_factor).unsqueeze(0).expand(B, -1, -1)
            map_y = (yy * distortion_factor).unsqueeze(0).expand(B, -1, -1)
            return torch.stack((map_x, map_y), dim=-1)
        def remap_channel(channel_tensor, custom_grid):
            channel_tensor = channel_tensor.permute(0, 3, 1, 2)
            return F.grid_sample(channel_tensor, custom_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        center_dist = torch.sqrt(xx**2 + yy**2)
        center_dist = center_dist / center_dist.max()
        distorted = {}
    
        if method in ['pincushion', 'barrel']:
            scale = 0.02 * strength
            if method == 'pincushion':
                distortion_red = 1 - (center_dist * scale * distort_channel_1_strength)
                distortion_green = 1 - (center_dist * scale * distort_channel_2_strength)
                distortion_blue = 1 - (center_dist * scale * distort_channel_3_strength)
            elif method == 'barrel':
                distortion_red = 1 + (center_dist * scale * distort_channel_1_strength)
                distortion_green = 1 + (center_dist * scale * distort_channel_2_strength)
                distortion_blue = 1 + (center_dist * scale * distort_channel_3_strength)
            if 'red' in distort_channels:
                red_grid = apply_distortion(distortion_red)
                red = remap_channel(image[:, :, :, 2:3], red_grid)
            else:
                red = image[:, :, :, 2:3].permute(0, 3, 1, 2)
            if 'green' in distort_channels:
                green_grid = apply_distortion(distortion_green)
                green = remap_channel(image[:, :, :, 1:2], green_grid)
            else:
                green = image[:, :, :, 1:2].permute(0, 3, 1, 2)
            if 'blue' in distort_channels:
                blue_grid = apply_distortion(distortion_blue)
                blue = remap_channel(image[:, :, :, 0:1], blue_grid)
            else:
                blue = image[:, :, :, 0:1].permute(0, 3, 1, 2)
    
        elif method == 'basic':
            red = torch.roll(image[:, :, :, 2], shifts=shift, dims=1).unsqueeze(1)
            green = image[:, :, :, 1].unsqueeze(1)
            blue = torch.roll(image[:, :, :, 0], shifts=-shift, dims=1).unsqueeze(1)
    
        elif method == 'advanced':
            red = torch.roll(image[:, :, :, 2], shifts=shift, dims=1).unsqueeze(1)
            green = torch.roll(image[:, :, :, 1], shifts=-shift//2, dims=2).unsqueeze(1)
            blue = torch.roll(image[:, :, :, 0], shifts=shift, dims=1).unsqueeze(1)
    
        else:
            return (image,)
        final = torch.cat([blue, green, red], dim=1).permute(0, 2, 3, 1)
        if crop:
            gray = final.mean(dim=-1)
            non_black_mask = gray > 0.01
            coords = non_black_mask.nonzero(as_tuple=False)
    
            if coords.numel() > 0:
                top = coords[:, 1].min().item()
                bottom = coords[:, 1].max().item()
                left = coords[:, 2].min().item()
                right = coords[:, 2].max().item()
                feather = int(((W / 1280 + H / 960) / 2) * ((strength * 10) * 2 + 2))
                top = max(top + feather, 0)
                bottom = min(bottom - feather, H - 1)
                left = max(left + feather, 0)
                right = min(right - feather, W - 1)
                final = final[:, top:bottom + 1, left:right + 1, :]
    
        return (final,)
        
class Lens_Distortion:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005}),
                "method": (
            [   
                'barrel',
                'pincushion',
            ], {
                "default": 'barrel'
            }),
                "crop": ("BOOLEAN", {"default": True}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_lens_distortion"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding Lens Distortion to an image. It has barrel and pincushion methods. Use crop for barrel distortion.

"""

    def func_lens_distortion(self, image, strength, method, crop):
        B, H, W, C = image.shape
        device = image.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        xx = xx.expand(B, -1, -1)
        yy = yy.expand(B, -1, -1)
        r2 = xx**2 + yy**2
        if method == 'barrel':
            distortion = 1 + strength * r2
        elif method == 'pincushion':
            distortion = 1 - strength * r2
        else:
            distortion = torch.ones_like(r2)
        map_x = xx * distortion
        map_y = yy * distortion
        grid = torch.stack((map_x, map_y), dim=-1)
        distorted = image.permute(0, 3, 1, 2)
        distorted = F.grid_sample(distorted, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        distorted = distorted.permute(0, 2, 3, 1)
        if crop:
            valid_mask = (
                (map_x >= -1) & (map_x <= 1) &
                (map_y >= -1) & (map_y <= 1)
            ).float()
            cropped_images = []
            for b in range(B):
                mask = valid_mask[b]
                coords = mask.nonzero(as_tuple=False)
                if coords.shape[0] == 0:
                    cropped_images.append(distorted[b:b+1])
                    continue
                y_min = coords[:, 0].min().item()
                y_max = coords[:, 0].max().item()
                x_min = coords[:, 1].min().item()
                x_max = coords[:, 1].max().item()
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2
                y1, y2 = 0, H
                x1, x2 = 0, W
                iteration = 0
                while True:
                    h = y2 - y1
                    w = x2 - x1
                    scale = 0.99
                    new_y1 = int(center_y - (h * scale) / 2)
                    new_y2 = int(center_y + (h * scale) / 2)
                    new_x1 = int(center_x - (w * scale) / 2)
                    new_x2 = int(center_x + (w * scale) / 2)
                    crop_mask = mask[new_y1:new_y2, new_x1:new_x2]
                    if crop_mask.min() < 1:
                        y1, y2 = new_y1, new_y2
                        x1, x2 = new_x1, new_x2
                    else:
                        break
                    if (y2 - y1 <= 1 or x2 - x1 <= 1):
                        break
                    iteration += 1
                shrink = 0.992
                final_h = y2 - y1
                final_w = x2 - x1
                y1 = int(center_y - (final_h * shrink) / 2)
                y2 = int(center_y + (final_h * shrink) / 2)
                x1 = int(center_x - (final_w * shrink) / 2)
                x2 = int(center_x + (final_w * shrink) / 2)
                cropped = distorted[b:b+1, y1:y2, x1:x2, :]
                cropped_images.append(cropped)
            return (torch.cat(cropped_images, dim=0),)
        else:
            return (distorted,)
            
class Depth_of_Field:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "blur_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.005}),
                "blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.005}),
                "dof_mask_size": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.005}),
                "falloff_size": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.005}),
                "sigma": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "mode": (
            [   
                'eliptical dof',
                'use mask',
            ], {
                "default": 'eliptical dof'
            }),
            },
            "optional":
            {
                "dof_mask": ("MASK",),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_dof_blur"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding depth of field to an image. You can supply a dof mask (white = in focus) from a depth map or just use the eliptical mode.

"""

    def func_dof_blur(self, image, blur_strength, blur_sigma, dof_mask_size, falloff_size, sigma, mode, dof_mask=None):
        blur_strength = blur_strength * 10
        blur_sigma = blur_sigma * 10
        B, H, W, C = image.shape
        device = image.device
        def gaussian_1d_kernel(kernel_size, blur_sigma):
            x = torch.arange(kernel_size, device=device) - kernel_size // 2
            k = torch.exp(-0.5 * (x / blur_sigma) ** 2)
            k /= k.sum()
            return k
        def apply_separable_blur(img, kernel_size, blur_sigma):
            img = img.permute(0, 3, 1, 2)
            k1d = gaussian_1d_kernel(kernel_size, blur_sigma)
            kx = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
            pad = kernel_size // 2
            img = F.pad(img, (pad, pad, 0, 0), mode='reflect')
            img = F.conv2d(img, kx, groups=C)
            ky = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
            img = F.pad(img, (0, 0, pad, pad), mode='reflect')
            img = F.conv2d(img, ky, groups=C)
            return img.permute(0, 2, 3, 1)
        def create_ellipse_mask():
            y = torch.linspace(0, H - 1, H, device=device)
            x = torch.linspace(0, W - 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            cy, cx = H / 2, W / 2
            a = W * dof_mask_size
            b = H * dof_mask_size
            d = ((xx - cx) ** 2 / a**2) + ((yy - cy) ** 2 / b**2)
            mask = torch.exp(-d**(falloff_size * 10) / ((falloff_size * 10) * sigma**(falloff_size * 10)))
            mask = mask.clamp(0, 1)
            return mask.unsqueeze(0).expand(B, H, W)
        kernel_size = max(3, int(blur_strength * 10) | 1)
        blurred = apply_separable_blur(image, kernel_size, blur_sigma)
        if mode == "eliptical dof":
            mask = create_ellipse_mask()
        elif mode == "use mask":
            assert dof_mask is not None, "External dof_mask must be provided in mode=2"
            if dof_mask.shape[-2:] != (H, W):
                dof_mask = dof_mask.unsqueeze(1) if dof_mask.dim() == 3 else dof_mask
                dof_mask = F.interpolate(dof_mask, size=(H, W), mode="bicubic", align_corners=False)
                dof_mask = dof_mask.squeeze(1)
            mask = dof_mask.clamp(0, 1)
        else:
            raise ValueError("Unsupported mode. Use 1 or 2.")
        mask = mask.unsqueeze(-1)
        return (image * mask + blurred * (1 - mask),)
        
class Sensor_Dust:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "max_dust_factor": ("FLOAT", {"default": 0.0000005, "min": 0.0, "max": 0.0020000, "step": 0.0000005}),
                "minimum_number_spots": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
                "brightness_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.005}),
                "red_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
                "green_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
                "blue_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_sensor_dust"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding camera sensor dust spots to an image. max_dust_factor is this, @ 5e-7 =~ 12 spots on a 6000x4000px image. If your input image res >= 2000 in any dim it applies a different method. I noticed this many spots on my camera and wanted to emulate it here. These spots are usually a 3x3px grid of either red green or blue spots that are desaturated.

"""

    def func_sensor_dust(self, image, max_dust_factor, minimum_number_spots, intensity, brightness_threshold, red_intensity, green_intensity, blue_intensity):
        B, H, W, C = image.shape
        assert C == 3, "Expected last dimension to be RGB channels"
        assert image.dtype == torch.float32, "Image must be float32"
        device = image.device
        output = image.clone()
        is_high_res = H > 2000 or W > 2000
        total_pixels = H * W
        max_dust = int(total_pixels * max_dust_factor)
        color_weights = []
        if red_intensity > 0:
            color_weights.extend([0] * int(10 * red_intensity))
        if green_intensity > 0:
            color_weights.extend([1] * int(10 * green_intensity))
        if blue_intensity > 0:
            color_weights.extend([2] * int(10 * blue_intensity))
        color_weights = torch.tensor(color_weights if color_weights else [0], device=device)
        for b in range(B):
            num_spots = torch.randint(0, max(max_dust, 1), (1,), device=device).item()
            if num_spots == 0:
                num_spots = minimum_number_spots
            blocked = torch.zeros((H, W), dtype=torch.bool, device=device)
            for _ in range(num_spots):
                for _ in range(100):
                    if is_high_res:
                        y = torch.randint(0, H - 1, (1,), device=device).item()
                        x = torch.randint(0, W - 1, (1,), device=device).item()
                        coords = [(y + i, x + j) for i in range(2) for j in range(2)]
                    else:
                        y = torch.randint(0, H, (1,), device=device).item()
                        x = torch.randint(0, W, (1,), device=device).item()
                        coords = [(y, x)]
                    if all(0 <= cy < H and 0 <= cx < W and not blocked[cy, cx] for cy, cx in coords):
                        break
                brightness = (0.299 * output[b, y, x, 0] +
                            0.587 * output[b, y, x, 1] +
                            0.114 * output[b, y, x, 2])
                dust_factor = 1.0 if brightness <= brightness_threshold else 0.5
                bleed_channel = color_weights[torch.randint(0, len(color_weights), (1,), device=device)].item()
                base_intensity = (torch.randint(50, 200, (1,), device=device).float().item() / 255.0) * intensity * dust_factor
                if is_high_res:
                    for (cy, cx) in coords:
                        for c in range(3):
                            output[b, cy, cx, c] += base_intensity if c == bleed_channel else base_intensity * 0.5
                        blocked[cy, cx] = True
                    for dy in range(-7, 9):
                        for dx in range(-7, 9):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                blocked[ny, nx] = True
                    for dy in range(-1, 3):
                        for dx in range(-1, 3):
                            ny, nx = y + dy, x + dx
                            if (y <= ny <= y + 1) and (x <= nx <= x + 1):
                                continue
                            if 0 <= ny < H and 0 <= nx < W:
                                neighbor = abs(dy) + abs(dx) == 1
                                factor = 0.0625 if neighbor else 0.0225
                                bleed = (torch.randint(90, 100, (1,), device=device).float().item() / 255.0) * factor * dust_factor
                                output[b, ny, nx, bleed_channel] += bleed
                else:
                    for c in range(3):
                        output[b, y, x, c] += base_intensity if c == bleed_channel else base_intensity * 0.5
                    for dy in range(-7, 8):
                        for dx in range(-7, 8):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                blocked[ny, nx] = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and not (dy == 0 and dx == 0):
                                factor = 0.0625 if abs(dx) + abs(dy) == 1 else 0.0225
                                bleed = (torch.randint(90, 100, (1,), device=device).float().item() / 255.0) * factor * dust_factor
                                output[b, ny, nx, bleed_channel] += bleed
        return (torch.clamp(output, 0.0, 1.0),)
        
class Lens_Dirt:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "dirt_path": ("STRING", {"default": ""}),
                "crop_to_aspect": ("BOOLEAN", {"default": True}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional":
            {
                "dirt_texture": ("IMAGE",),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_lens_dirt"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding lens dirt to an image. Think like reshade, get some dust or dirt texture where black is not where dirt is.

dirt_path = path to where the texture is or use the optional dirt_texture input
crop_to_aspect = both rotate the dirt tex to your input image aspect then crop to the same size
intensity = intensity of the effect
opacity = like in photoshop opacity amount

"""

    def func_lens_dirt(self, image, dirt_path, crop_to_aspect, intensity, opacity, dirt_texture=None):
        opacity = opacity * 0.01
        assert image.ndim == 4 and image.dtype == torch.float32
        assert 0.0 <= intensity <= 1.0
        assert 0.0 <= opacity <= 1.0
        dirt_path = dirt_path.strip('\'"')
        if dirt_texture == None:
            dirt_tex = read_image(dirt_path).float() / 255.0
            dirt_tex = dirt_tex.permute(1, 2, 0).unsqueeze(0)
        else:
            dirt_tex = dirt_texture
        B, H, W, C = image.shape
        _, H_d, W_d, _ = dirt_tex.shape
        if crop_to_aspect:
            target_aspect = W / H
            dirt_aspect = W_d / H_d
            if (dirt_aspect > 1 and target_aspect < 1) or (dirt_aspect < 1 and target_aspect > 1):
                dirt_tex = dirt_tex.transpose(1, 2).flip(2)
                H_d, W_d = W_d, H_d
            dirt_aspect = W_d / H_d
            if dirt_aspect > target_aspect:
                new_width = int(H_d * target_aspect)
                x0 = (W_d - new_width) // 2
                dirt_tex = dirt_tex[:, :, x0:x0 + new_width, :]
            else:
                new_height = int(W_d / target_aspect)
                y0 = (H_d - new_height) // 2
                dirt_tex = dirt_tex[:, y0:y0 + new_height, :, :]
        dirt_tex = dirt_tex.permute(0, 3, 1, 2)
        dirt_tex = F.interpolate(dirt_tex, size=(H, W), mode='bilinear', align_corners=False)
        dirt_tex = dirt_tex.permute(0, 2, 3, 1)
        if dirt_tex.shape[0] == 1 and B > 1:
            dirt_tex = dirt_tex.expand(B, -1, -1, -1)
        dirt_tex = dirt_tex * intensity
        screen = 1.0 - (1.0 - image) * (1.0 - dirt_tex)
        blended = image * (1 - opacity) + screen * opacity
        return (blended.clamp(0.0, 1.0),)
        
class Physically_Accurate_Lens_Dirt:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "dirt_texture": ("IMAGE",),
                "bokeh_texture": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.005}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "dof_kernel_size": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "dirt_scale": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_physically_accurate_lens_dirt"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding "physically accurate" lens dirt to an image

intensity = intensity of the dirt texture to blend on top of the origional image
threshold = Where the dirt will show up, like the max white point
dof_kernel_size = blur of the dirt texture, if you are using a reshade texture you may not need to blur
dirt_scale = scale of the dirt texture, if the spots art to small scale it up, if using a reshade texture set to 1

"""

    def func_physically_accurate_lens_dirt(self, image, intensity, threshold, dirt_texture, bokeh_texture, dof_kernel_size, dirt_scale):
        def fft_convolve2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
            B, C, H, W = input.shape
            _, _, Hk, Wk = kernel.shape
            pad_h = H - Hk
            pad_w = W - Wk
            kernel_padded = F.pad(kernel, (0, pad_w, 0, pad_h))
            input_fft = torch.fft.rfftn(input, s=(H, W), dim=(-2, -1))
            kernel_fft = torch.fft.rfftn(kernel_padded, s=(H, W), dim=(-2, -1))
            output_fft = input_fft * kernel_fft
            output = torch.fft.irfftn(output_fft, s=(H, W), dim=(-2, -1))
            return output
        def generate_aperture_kernel(shape: str, Hk: int, Wk: int) -> torch.Tensor:
            y = torch.linspace(-1.0, 1.0, Hk)
            x = torch.linspace(-1.0, 1.0, Wk)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            if shape == 'circle':
                mask = (xx**2 + yy**2) <= 1.0
            elif shape == 'square':
                mask = (xx.abs() <= 1.0) & (yy.abs() <= 1.0)
            else:
                raise ValueError(f"Unsupported aperture shape: {shape}")
            kernel = mask.float().unsqueeze(0).unsqueeze(0)
            kernel /= (kernel.sum() + 1e-12)
            return kernel
        def apply_lens_dirt(
            image: torch.Tensor,
            dirt_texture: torch.Tensor,
            bokeh_texture: torch.Tensor = None,
            aperture_shape: str = None,
            scattering_spectrum: torch.Tensor = None,
            threshold: float = 0.8,
            intensity: float = 1.0,
            dof_kernel_size: int = 128,
            dirt_scale: int = 4
        ) -> torch.Tensor:
            B, H, W, C = image.shape
            img = image.permute(0, 3, 1, 2).contiguous()
            luma = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            highlight_mask = torch.clamp((luma - threshold) / (1.0 - threshold), min=0.0)
            if aperture_shape is not None:
                Hk = bokeh_texture.shape[1] if bokeh_texture is not None else 64
                Wk = bokeh_texture.shape[2] if bokeh_texture is not None else 64
                bk = generate_aperture_kernel(aperture_shape, Hk, Wk)
                bk = bk.repeat(1, C, 1, 1)
            else:
                bk = bokeh_texture
                if bk.dim() == 4:
                    bk = bk.permute(0, 3, 1, 2)
                else:
                    bk = bk.unsqueeze(1)
                bk = bk.to(torch.float32)
                bk = bk / (bk.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            highlight_bokeh = fft_convolve2d(highlight_mask, bk)
            dirt = dirt_texture
            if dirt.dim() == 4:
                dirt = dirt.permute(0, 3, 1, 2)
            else:
                dirt = dirt.unsqueeze(1)
            _, _, H_img, W_img = img.shape
            _, _, H_dirt, W_dirt = dirt.shape
            img_is_landscape = W_img >= H_img
            dirt_is_landscape = W_dirt >= H_dirt
            if img_is_landscape != dirt_is_landscape:
                dirt = torch.rot90(dirt, k=1, dims=[-2, -1])
            dirt_resized = dirt
            if dof_kernel_size > 0:
                dof_k = generate_aperture_kernel(aperture_shape or 'circle', dof_kernel_size, dof_kernel_size)
                dof_k = dof_k.repeat(1, C, 1, 1)
                dirt_resized = fft_convolve2d(dirt_resized, dof_k)
            H_up, W_up = H * dirt_scale, W * dirt_scale
            dirt_up = F.interpolate(dirt_resized, size=(H_up, W_up), mode='bilinear', align_corners=False)
            start_y = (H_up - H) // 2
            start_x = (W_up - W) // 2
            dirt_resized = dirt_up[:, :, start_y:start_y + H, start_x:start_x + W]
            if scattering_spectrum is not None:
                spec = scattering_spectrum.view(1, C, 1, 1).to(highlight_bokeh)
                highlight_bokeh = highlight_bokeh * spec
            scattered = highlight_bokeh * dirt_resized
            out = img + scattered * intensity
            out = torch.clamp(out, 0.0, 1.0)
            return out.permute(0, 2, 3, 1).contiguous()
        spectrum = torch.tensor([1.0, 0.8, 0.6])
        output = apply_lens_dirt(image, dirt_texture, bokeh_texture, aperture_shape='square', scattering_spectrum=spectrum, threshold=threshold, intensity=intensity, dof_kernel_size=dof_kernel_size, dirt_scale=dirt_scale)
        return (output,)

class Bloom_Lens_Flares:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "tex_dirt": ("IMAGE",),
                "tex_sprite": ("IMAGE",),
                "enable_bloom": ("BOOLEAN", {"default": True}),
                "bloom_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "bloom_amount": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.005}),
                "bloom_saturation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.005}),
                "bloom_tint_r": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.005}),
                "bloom_tint_g": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "bloom_tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "bloom_mixmode": ("INT", {"default": 2, "min": 0, "max": 2, "step": 1}),
                "enable_lensdirt": ("BOOLEAN", {"default": True}),
                "lensdirt_intensity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.005}),
                "lensdirt_saturation": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.005}),
                "lensdirt_tint_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "lensdirt_tint_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "lensdirt_tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "lensdirt_mixmode": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "enable_anam_flare": ("BOOLEAN", {"default": True}),
                "anam_flare_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.005}),
                "anam_flare_amount": ("FLOAT", {"default": 14.5, "min": 0.0, "max": 40.0, "step": 0.05}),
                "anam_flare_curve": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.005}),
                "anam_flare_color_r": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 1.0, "step": 0.005}),
                "anam_flare_color_g": ("FLOAT", {"default": 0.313, "min": 0.0, "max": 1.0, "step": 0.005}),
                "anam_flare_color_b": ("FLOAT", {"default": 0.588, "min": 0.0, "max": 1.0, "step": 0.005}),
                "anam_flare_blur_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.005}),
                "anam_flare_blur_kernel_size": ("INT", {"default": 17, "min": 1, "max": 127, "step": 2}),
                "enable_lenz": ("BOOLEAN", {"default": True}),
                "lenz_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "lenz_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "enable_chap_flare": ("BOOLEAN", {"default": True}),
                "chap_flare_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.005}),
                "chap_flare_count": ("INT", {"default": 15, "min": 1, "max": 127, "step": 1}),
                "chap_flare_dispersal": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.005}),
                "chap_flare_size": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.005}),
                "chap_flare_ca_r": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chap_flare_ca_g": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chap_flare_ca_b": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chap_flare_intensity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "enable_godray": ("BOOLEAN", {"default": True}),
                "godray_decay": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.005}),
                "godray_exposure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.005}),
                "godray_weight": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 1.5, "step": 0.005}),
                "godray_density": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.005}),
                "godray_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.005}),
                "godray_samples": ("INT", {"default": 128, "min": 1, "max": 1024, "step": 1}),
                "enable_flare_color": ("BOOLEAN", {"default": True}),
                "flare_luminance": ("FLOAT", {"default": 0.095, "min": 0.0, "max": 1.0, "step": 0.005}),
                "flare_blur": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 400.0, "step": 0.5}),
                "flare_intensity": ("FLOAT", {"default": 2.07, "min": 0.0, "max": 4.0, "step": 0.05}),
                "flare_tint_r": ("FLOAT", {"default": 0.137, "min": 0.0, "max": 1.0, "step": 0.001}),
                "flare_tint_g": ("FLOAT", {"default": 0.216, "min": 0.0, "max": 1.0, "step": 0.001}),
                "flare_tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_bloom_and_lens_flares"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding Bloom and Lens effects to an image. // Bloom.fx - Copyright (c) 2009-2015 Gilcher Pascal aka Marty McFly. Modified by me to work with torch

"""

    def func_bloom_and_lens_flares(self,
        image: torch.Tensor,
        tex_dirt: torch.Tensor,
        tex_sprite: torch.Tensor,
        enable_bloom: bool = False,
        bloom_threshold: float = 0.8,
        bloom_amount: float = 0.8,
        bloom_saturation: float = 0.8,
        bloom_tint_r: float = 0.7,
        bloom_tint_g: float = 0.8,
        bloom_tint_b: float = 1.0,
        bloom_mixmode: int = 2,
        enable_lensdirt: bool = False,
        lensdirt_intensity: float = 0.4,
        lensdirt_saturation: float = 2.0,
        lensdirt_tint_r: float = 1.0,
        lensdirt_tint_g: float = 1.0,
        lensdirt_tint_b: float = 1.0,
        lensdirt_mixmode: int = 1,
        enable_anam_flare: bool = False,
        anam_flare_threshold: float = 0.9,
        anam_flare_amount: float = 14.5,
        anam_flare_curve: float = 1.2,
        anam_flare_color_r: float = 0.012,
        anam_flare_color_g: float = 0.313,
        anam_flare_color_b: float = 0.588,
        anam_flare_blur_strength: float = 0.9,
        anam_flare_blur_kernel_size: int = 17,
        enable_lenz: bool = False,
        lenz_threshold: float = 0.8,
        lenz_intensity: float = 1.0,
        enable_chap_flare: bool = False,
        chap_flare_threshold: float = 0.8,
        chap_flare_count: int = 15,
        chap_flare_dispersal: float = 0.25,
        chap_flare_size: float = 0.45,
        chap_flare_ca_r: float = 0.00,
        chap_flare_ca_g: float = 0.01,
        chap_flare_ca_b: float = 0.02,
        chap_flare_intensity: float = 100.0,
        enable_godray: bool = False,
        godray_decay: float = 0.99,
        godray_exposure: float = 1.0,
        godray_weight: float = 1.25,
        godray_density: float = 1.0,
        godray_threshold: float = 0.9,
        godray_samples: int = 128,
        enable_flare_color: bool = False,
        flare_luminance: float = 0.095,
        flare_blur: float = 200.0,
        flare_intensity: float = 2.07,
        flare_tint_r: float = 0.137,
        flare_tint_g: float = 0.216,
        flare_tint_b: float = 1.0,
    ) -> torch.Tensor:
        B, H, W, C = image.shape
        device = image.device
        bloom_tint = (bloom_tint_r, bloom_tint_g, bloom_tint_b)
        lensdirt_tint = (lensdirt_tint_r, lensdirt_tint_g, lensdirt_tint_b)
        anam_flare_color = (anam_flare_color_r, anam_flare_color_g, anam_flare_color_b)
        chap_flare_ca = (chap_flare_ca_r, chap_flare_ca_g, chap_flare_ca_b)
        flare_tint = (flare_tint_r, flare_tint_g, flare_tint_b)
        def match_tex(tex):
            _, Ht, Wt, _ = tex.shape
            ar_img = W / H
            ar_tex = Wt / Ht
            if abs(ar_tex - ar_img) > abs((1/ar_tex) - ar_img):
                tex = tex.permute(0,2,1,3).flip(2)
                Ht, Wt = Wt, Ht
                ar_tex = Wt / Ht
            scale = max(W / Wt, H / Ht)
            new_H, new_W = int(Ht * scale), int(Wt * scale)
            t = tex.permute(0,3,1,2)
            t = F.interpolate(t, size=(new_H, new_W), mode='bilinear', align_corners=False)
            tex = t.permute(0,2,3,1)
            off_h = (new_H - H) // 2
            off_w = (new_W - W) // 2
            return tex[:, off_h:off_h+H, off_w:off_w+W, :]
        tex_dirt  = match_tex(tex_dirt)
        tex_sprite= match_tex(tex_sprite)
        b_tint = torch.tensor(bloom_tint, device=device).view(1,1,1,3)
        ld_tint = torch.tensor(lensdirt_tint, device=device).view(1,1,1,3)
        af_color = torch.tensor(anam_flare_color, device=device).view(1,1,1,3)
        cf_ca    = torch.tensor(chap_flare_ca, device=device).view(1,1,1,3)
        f_tint   = torch.tensor(flare_tint, device=device).view(1,1,1,3)
        gw = torch.tensor(
            [0.016216,0.054054,0.121622,0.194595,0.227027,
            0.194595,0.121622,0.054054,0.016216],
            device=device, dtype=torch.float32
        ).view(1,1,-1,1)
        def blur1d(t):
            B_, C_, H_, W_ = t.shape
            w1 = gw.repeat(C_, 1, 1, 1)
            v = F.conv2d(t, w1, padding=(4,0), groups=C_)
            w2 = w1.transpose(2,3)
            return F.conv2d(v, w2, padding=(0,4), groups=C_)
        def sample_uv(tex, uv):
            t = tex.permute(0,3,1,2)
            return F.grid_sample(t, uv, mode='bilinear', align_corners=False) \
                .permute(0,2,3,1)
        def get_distorted_tex(x, center_uv, sample_vec, ca):
            uv_r = center_uv + sample_vec * ca[..., 0:1]
            uv_g = center_uv + sample_vec * ca[..., 1:2]
            uv_b = center_uv + sample_vec * ca[..., 2:3]
            r = sample_uv(x[..., 0:1], uv_r)
            g = sample_uv(x[..., 1:2], uv_g)
            b = sample_uv(x[..., 2:3], uv_b)
            return torch.cat((r, g, b), dim=-1)
        aspect = float(W) / float(H)
        ys = torch.linspace(-1,1,H,device=device)
        xs = torch.linspace(-1,1,W,device=device)
        yy, xx = torch.meshgrid(ys,xs, indexing='ij')
        base_uv = torch.stack((xx,yy),-1).unsqueeze(0).repeat(B,1,1,1)
        center_uv = torch.zeros_like(base_uv)
        x = image
        bright_mask = F.relu(x.mean(-1, keepdim=True) - chap_flare_threshold)
        if enable_bloom:
            bright = F.relu(x - bloom_threshold)
            t0 = bright.permute(0,3,1,2)
            p1 = F.avg_pool2d(t0,2); p2 = F.avg_pool2d(p1,2); p3 = F.avg_pool2d(p2,2)
            b1 = blur1d(p1); b2 = blur1d(p2); b3 = blur1d(p3)
            u2 = F.interpolate(b3, scale_factor=2, mode='bilinear', align_corners=False) + b2
            u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False) + b1
            bloom = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
            bloom = bloom.permute(0,2,3,1)
            gray = bloom.mean(-1, keepdim=True)
            bloom = gray + (bloom - gray) * bloom_saturation
            bloom = bloom * b_tint * bloom_amount
            base = x
            if   bloom_mixmode == 0:  x = base + bloom
            elif bloom_mixmode == 1:  x = 1 - (1-base)*(1-bloom)
            else:                     x = torch.max(base, bloom)
        else:
            bloom = torch.zeros_like(x)
        if enable_anam_flare:
            def get_bright(coords):
                samp = sample_uv(x, coords)
                b = F.relu(samp - anam_flare_threshold)
                lum = b.mean(-1, keepdim=True)
                t = (lum / 0.5).clamp(0,1)
                return samp * t
            gw = torch.tensor([
                0.194595,
                0.227027,
                0.194595,
                0.121622,
                0.054054
            ], device=device)
            gauss = torch.cat([gw.flip(0), gw[1:]])
            step = 2.0 / H
            anam = torch.zeros_like(x)
            for i, w in enumerate(gauss):
                z = i - 4
                uv_z = base_uv + torch.tensor([0.0, z * step], device=device).view(1,1,1,2)
                coords = uv_z * 2 - 1
                coords = torch.stack([
                    -coords[...,0] / flare_blur,
                    coords[...,1]
                ], -1) * 0.5 + 0.5
                anam += get_bright(coords) * w
            anam = anam * af_color
            anam = anam.pow(1.0 / anam_flare_curve)
            anam_flare_blur_strength    = anam_flare_blur_strength
            anam_flare_blur_kernel_size = anam_flare_blur_kernel_size | 1
            anam_flare_blur_kernel_size = anam_flare_blur_kernel_size
            if anam_flare_blur_strength > 0 and anam_flare_blur_kernel_size > 1:
                streaks = anam.permute(0,3,1,2)
                blur_v = F.avg_pool2d(
                    streaks,
                    kernel_size=(anam_flare_blur_kernel_size, 1),
                    stride=1,
                    padding=(anam_flare_blur_kernel_size//2, 0),
                    count_include_pad=False
                )
                anam_blur = blur_v.permute(0,2,3,1)
                anam = anam.lerp(anam_blur, anam_flare_blur_strength)
            x = x + anam
        if enable_lensdirt:
            mask = bloom.mean(-1,keepdim=True)
            ld = tex_dirt * mask * lensdirt_intensity
            g = ld.mean(-1,keepdim=True)
            ld = g + (ld-g)*lensdirt_saturation
            if   lensdirt_mixmode == 0: x = x + ld
            elif lensdirt_mixmode == 1: x = 1 - (1-x)*(1-ld)
            else:                        x = x + ld
        lf = torch.zeros_like(x)
        center = torch.tensor([0.0,0.0],device=device)
        if enable_lenz:
            offsets = [(0.9,0.01,4),(0.7,0.25,25),(0.3,0.25,15),(1,1,5),
                    (-0.15,20,1),(-0.3,20,1),(6,6,6),(7,7,7),(8,8,8),(9,9,9),
                    (0.24,1,10),(0.32,1,10),(0.4,1,10),(0.5,-0.5,2),(2,2,-5),
                    (-5,0.2,0.2),(20,0.5,0),(0.4,1,10),(1e-5,10,20)]
            factors = [(1.5,1.5,0),(0,1.5,0),(0,0,1.5),(0.2,0.25,0),(0.15,0,0),(0,0,0.15),
                    (1.4,0,0),(1,1,0),(0,1,0),(0,0,1.4),(1,0.3,0),(1,1,0),(0,2,4),
                    (0.2,0.1,0),(0,0,1),(1,1,0),(1,1,0),(0,0,0.2),(0.012,0.313,0.588)]
            dist = base_uv - center
            dist_x = dist[...,0] * (W/H)
            dv = torch.stack((dist_x,dist[...,1]),-1)
            for (ox,py,sp),(fr,fg,fb) in zip(offsets,factors):
                r = torch.linalg.norm(dv,dim=-1,keepdim=True)
                uv = center + dv * ox * (2*r).pow(py*3.5) * sp
                samp = sample_uv(x, uv)
                w = (1 - (((uv-center)*2)**2).sum(-1,keepdim=True)).clamp(0,1)
                glow = F.relu(samp.mean(-1,keepdim=True)-lenz_threshold) * torch.tensor([fr,fg,fb],device=device)
                lf += glow * w * lenz_intensity
        if enable_chap_flare:
            sv = (center_uv - base_uv) * chap_flare_dispersal
            sv = torch.stack((sv[...,0] * aspect, sv[...,1]), dim=-1)
            norm = sv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            halo_vec = sv / norm * chap_flare_size
            chap = get_distorted_tex(
                x * bright_mask,
                base_uv + halo_vec,
                halo_vec,
                cf_ca * 2.5
            )
            for j in range(chap_flare_count):
                uv_offset = base_uv + sv * (j / float(chap_flare_count))
                chap += get_distorted_tex(
                    x * bright_mask,
                    uv_offset,
                    sv,
                    cf_ca
                )
            chap = chap / float(chap_flare_count + 1)
            chap = chap.clamp(0.0, 1.0)
            lf += chap * chap_flare_intensity
        if enable_godray:
            delta = (base_uv-center) * (1/godray_samples) * godray_density
            suv = base_uv.clone()
            decay = 1.0
            for _ in range(godray_samples):
                suv = suv - delta
                samp = sample_uv(x, suv)
                mask = F.relu(samp.mean(-1,keepdim=True)-godray_threshold)
                col = samp * torch.tensor([1.0,0.95,0.85],device=device).view(1,1,1,3)
                lf += col * mask * decay * godray_weight
                decay *= godray_decay
            lf *= godray_exposure
        if tex_sprite is not None:
            off = 1.0 / W
            m = 0
            for dx,dy in ((off,off),(-off,off),(off,-off),(-off,-off)):
                m = m + sample_uv(tex_sprite, base_uv + torch.tensor([dx,dy],device=device))
            mask = m * 0.25
            lf = lf * mask
        x = x + lf
        if enable_flare_color:
            br = F.relu(x - flare_luminance)
            t = br.permute(0,3,1,2)
            bb = blur1d(t)
            bb = blur1d(bb.transpose(2,3)).transpose(2,3)
            bb = F.interpolate(bb, size=(H, W), mode='bilinear', align_corners=False)
            bb = bb.permute(0,2,3,1) * f_tint * flare_intensity
            x = x + bb
        return (x,)

class Halation:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.05}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 15, "min": 0, "max": 88, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_halation"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding an Halation effect on an image.

"""

    def func_halation(self, image, threshold: float = 0.8, intensity: float = 0.5, blur_radius: int = 15):
        B, H, W, C = image.shape
        assert C == 3, "Expected image with 3 channels (RGB)"
        luma_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(1, 1, 1, 3)
        luminance = (image * luma_weights).sum(dim=-1, keepdim=True)
        halo_mask = torch.clamp((luminance - threshold) / (1.0 - threshold), 0.0, 1.0)
        halo_color = image * halo_mask
        halo_color = halo_color.clone()
        halo_color[..., 1:] *= 0.3
        halo_color = halo_color.permute(0, 3, 1, 2)
        def get_gaussian_kernel(radius: int, sigma: float):
            k = torch.arange(-radius, radius + 1, device=image.device).float()
            k = torch.exp(-0.5 * (k / sigma).pow(2))
            k /= k.sum()
            return k
        sigma = blur_radius / 3
        kernel = get_gaussian_kernel(blur_radius, sigma)
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.view(1, 1, *kernel_2d.shape)
        kernel_2d = kernel_2d.expand(C, 1, -1, -1)
        blurred = F.conv2d(halo_color, kernel_2d, padding=blur_radius, groups=C)
        blurred = F.conv2d(blurred, kernel_2d, padding=blur_radius, groups=C)
        blurred = blurred.permute(0, 2, 3, 1)
        output = torch.clamp(image + blurred * intensity, 0.0, 1.0)
        return (output,)

class Sharpen:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_sharpen"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding very basic sharpening to an image.

"""

    def func_sharpen(self, image, strength):
        assert image.dtype == torch.float32 and image.ndim == 4 and image.shape[-1] == 3, \
            "Input must be a float32 tensor of shape [B, H, W, 3]"
        B, H, W, C = image.shape
        image_chw = image.permute(0, 3, 1, 2)
        high_pass_kernel = torch.tensor([
            [[-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]]
        ], dtype=torch.float32, device=image.device)
        kernel = high_pass_kernel.expand(C, 1, 3, 3)
        high_pass = F.conv2d(image_chw, kernel, padding=1, groups=C)
        sharpened = image_chw + strength * high_pass
        sharpened = torch.clamp(sharpened, 0.0, 1.0)
        return (sharpened.permute(0, 2, 3, 1),)
        
class Sharpen_Unsharp_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_sharpen_unsharp_mask"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for adding sharpening to an image, like photoshop's unsharp mask filter.

"""

    def func_sharpen_unsharp_mask(self, image, radius, strength):
        assert image.dtype == torch.float32 and image.ndim == 4 and image.shape[-1] == 3, \
            "Input must be a float32 tensor of shape [B, H, W, 3]"
        B, H, W, C = image.shape
        image_chw = image.permute(0, 3, 1, 2)
        def gaussian_kernel1d(kernel_size, sigma):
            coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
            kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
            kernel /= kernel.sum()
            return kernel
        kernel_size = max(3, int(radius * 4) | 1)
        sigma = radius
        pad = kernel_size // 2
        kernel1d = gaussian_kernel1d(kernel_size, sigma).to(image.device)
        kernel1d_x = kernel1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        kernel1d_y = kernel1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        image_padded_x = F.pad(image_chw, (pad, pad, 0, 0), mode='reflect')
        blurred_x = F.conv2d(image_padded_x, kernel1d_x, groups=C)
        image_padded_y = F.pad(blurred_x, (0, 0, pad, pad), mode='reflect')
        blurred = F.conv2d(image_padded_y, kernel1d_y, groups=C)
        high_freq = image_chw - blurred
        sharpened = image_chw + strength * high_freq
        sharpened = torch.clamp(sharpened, 0.0, 1.0)
        return (sharpened.permute(0, 2, 3, 1),)
        
class Manga_Toner:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "num_zones": ("INT", {"default": 5, "min": 3, "max": 10, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.05}),
                "edge_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005}),
                "edge_dilate": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_manga_toner"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying a simple manga toner effect to a image.

"""

    def func_manga_toner(self, image: torch.Tensor, num_zones: int = 5, spacings: list = None, radii: list = None, scale_factor: float = 2.0, edge_strength: float = 1.0, edge_threshold: float = 0.08, edge_dilate: int = 1):
        img = image.permute(0, 3, 1, 2)
        B, C, H, W = img.shape
        device = img.device
        gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2])
        thresholds = torch.linspace(0, 1, num_zones + 1, device=device)
        if spacings is None:
            spacings = [4 + i * 2 for i in range(num_zones)]
        if radii is None:
            radii = [max(0.5, 2.5 - i * 0.4) for i in range(num_zones)]
        if len(spacings) != num_zones:
            spacings = (spacings * ((num_zones + len(spacings) - 1) // len(spacings)))[:num_zones]
        if len(radii) != num_zones:
            radii = (radii * ((num_zones + len(radii) - 1) // len(radii)))[:num_zones]
        H_scaled, W_scaled = int(H * scale_factor), int(W * scale_factor)
        img_scaled = F.interpolate(img, size=(H_scaled, W_scaled), mode='bilinear', align_corners=False)
        gray_scaled = (0.299 * img_scaled[:, 0] + 0.587 * img_scaled[:, 1] + 0.114 * img_scaled[:, 2])
        out_scaled = torch.ones_like(gray_scaled)
        def generate_dot_pattern(h, w, spacing, radius):
            y, x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            dot_centers = ((y % spacing == 0) & (x % spacing == 0))
            pattern = torch.zeros((h, w), device=device)
            for dy in range(-math.ceil(radius), math.ceil(radius) + 1):
                for dx in range(-math.ceil(radius), math.ceil(radius) + 1):
                    if dx**2 + dy**2 <= radius**2:
                        shifted = torch.roll(torch.roll(dot_centers.float(), dx, dims=1), dy, dims=0)
                        pattern = torch.maximum(pattern, shifted)
            return pattern
        for i in range(num_zones):
            low, high = thresholds[i], thresholds[i + 1]
            mask = ((gray_scaled >= low) & (gray_scaled < high)).float()
            spacing = spacings[i]
            radius = radii[i]
            pat = 1.0 - generate_dot_pattern(H_scaled, W_scaled, spacing, radius)
            out_scaled = out_scaled * (1 - mask) + pat.unsqueeze(0) * mask
        def sobel_edges(img_gray, threshold=0.2, dilate=1):
            kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            gray_batched = img_gray.unsqueeze(1)
            grad_x = F.conv2d(gray_batched, kx, padding=1)
            grad_y = F.conv2d(gray_batched, ky, padding=1)
            mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            mag_flat = mag.view(mag.shape[0], -1)
            mag_max = mag_flat.max(dim=1)[0].view(-1, 1, 1, 1)
            mag_norm = mag / (mag_max + 1e-6)
            edges = (mag_norm > threshold).float()
            if dilate > 0:
                kernel = torch.ones((1, 1, 2 * dilate + 1, 2 * dilate + 1), device=device)
                edges = F.max_pool2d(edges, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
            return edges.squeeze(1)
        edges = sobel_edges(gray_scaled, threshold=edge_threshold, dilate=edge_dilate)
        final_scaled = out_scaled * (1.0 - edges * edge_strength)
        out_rgb = final_scaled.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0)
        return (out_rgb.permute(0, 2, 3, 1),)

class Monitor_Filter:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "pattern": ("IMAGE",),
                "cell_size": ("INT", {"default": 12, "min": 2, "max": 100, "step": 1}),
                "pattern_tint_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "pattern_tint_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "pattern_tint_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "glow_color_r": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.005}),
                "glow_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "glow_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "glow_offset_x": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "glow_offset_y": ("INT", {"default": 0, "min": -30, "max": 30, "step": 1}),
                "glow_blur_radius": ("INT", {"default": 13, "min": 1, "max": 55, "step": 2}),
                "glow_blur_sigma": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 99.0, "step": 0.05}),
                "glow_layer_2": ("BOOLEAN", {"default": True}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_monitor_filter"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying a cool pattern to an image that gets applied via hard mix blending. This gives the effect of a low resolution monitor or really anything you want with custom patterns.

"""

    def func_monitor_filter(self, image: torch.Tensor, pattern: torch.Tensor = None, cell_size: int = 12, pattern_tint_r: float = 1.0, pattern_tint_g: float = 1.0, pattern_tint_b: float = 1.0, glow_color_r: float = 0.1, glow_color_g: float = 1.0, glow_color_b: float = 0.0, glow_offset_x: int = 0, glow_offset_y: int = 0, glow_blur_radius: int = 13, glow_blur_sigma: float = 5.0, glow_layer_2: bool = True):
        def gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
            coords = torch.arange(kernel_size, device=image.device) - kernel_size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g = g / g.sum()
            g = g.to(dtype=image.dtype)
            g1d = g.view(1, 1, kernel_size)
            kernel_v = g1d.unsqueeze(3)
            kernel_h = g1d.unsqueeze(2)
            B, C, H, W = image.shape
            padding = kernel_size // 2
            image = F.conv2d(image, kernel_v.expand(C, 1, kernel_size, 1), padding=(padding, 0), groups=C)
            image = F.conv2d(image, kernel_h.expand(C, 1, 1, kernel_size), padding=(0, padding), groups=C)
            return image

        pattern_color_tint: tuple = (pattern_tint_r, pattern_tint_g, pattern_tint_b)
        pattern_color_tint_tensor = torch.tensor(pattern_color_tint, dtype=torch.float32, device=image.device)
        glow_color: tuple = (glow_color_r, glow_color_g, glow_color_b)
        glow_color_2: tuple = (glow_color_r - 0.2, glow_color_g - 0.2, glow_color_b - 0.2)
        glow_offset: tuple = (glow_offset_x, glow_offset_y)
        glow_blur_radius: int = glow_blur_radius
        glow_blur_radius = glow_blur_radius | 1
        glow_blur_sigma: float = glow_blur_sigma
        B, H, W, C = image.shape
        x = image.permute(0,3,1,2)
        x = F.avg_pool2d(x, kernel_size=cell_size, stride=cell_size)
        x = F.interpolate(x, size=(H, W), mode="nearest")
        pix = x.permute(0,2,3,1)
        p = pattern.permute(0,3,1,2)
        p = F.interpolate(p, size=(cell_size,cell_size), mode="bilinear", align_corners=False)
        pat_cell = p.permute(0,2,3,1)
        repeats_y = (H + cell_size - 1) // cell_size
        repeats_x = (W + cell_size - 1) // cell_size
        pattern_tiled = pat_cell.repeat(1, repeats_y, repeats_x, 1)
        pattern_tiled = pattern_tiled[:, :H, :W, :]
        out_color = torch.where(
            pix + pattern_tiled < 1.0,
            torch.zeros_like(pix),
            torch.ones_like(pix)
        )
        Y = (out_color[...,0]*1 +
            out_color[...,1]*1 +
            out_color[...,2]*1)
        out_bw = Y.unsqueeze(-1).expand(-1, -1, -1, C)
        mask = (out_bw[...,0] > 0.5).float().unsqueeze(1)
        dx, dy = glow_offset
        pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))
        shadow = F.pad(mask, pad=pad)[:, :, :H, :W]
        glow_col = torch.tensor(glow_color, device=image.device, dtype=image.dtype)
        shadow_rgb = shadow.repeat(1, C, 1, 1) * glow_col.view(1, C, 1, 1)
        blurred_glow = gaussian_blur(shadow_rgb, kernel_size=glow_blur_radius, sigma=glow_blur_sigma)
        if glow_layer_2:
            glow_2_col = torch.tensor(glow_color_2, device=image.device, dtype=image.dtype)
            shadow_2_rgb = shadow.repeat(1, C, 1, 1) * glow_2_col.view(1, C, 1, 1)
            blurred_glow_2 = gaussian_blur(shadow_2_rgb, kernel_size=glow_blur_radius + 6, sigma=glow_blur_sigma + 2.0)
            mask_fg = (out_bw.permute(0,3,1,2) > 0).float()
            fg_tinted = mask_fg * pattern_color_tint_tensor.view(1, C, 1, 1)
            comp = torch.clamp(blurred_glow + blurred_glow_2 + fg_tinted, 0.0, 1.0)
        else:
            mask_fg = (out_bw.permute(0,3,1,2) > 0).float()
            fg_tinted = mask_fg * pattern_color_tint_tensor.view(1, C, 1, 1)
            comp = torch.clamp(blurred_glow + fg_tinted, 0.0, 1.0)
        return (comp.permute(0,2,3,1),)

class VHS_Degrade:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "chroma_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "jitter": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "head_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "scanlines": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "dropouts": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_vhs_degrade"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying VHS degradation effects to an image.

chroma_shift = classic vhs chroma smear amount
jitter = vertical waves
head_noise = pixel noise at the bottom of the image area
scanlines = scanline intensity
dropouts = horizontal pixel strips of color noise
blur = blurs the effects + image

"""

    def func_vhs_degrade(self, image: torch.Tensor, chroma_shift = 1.0, jitter = 1.0, head_noise = 1.0, scanlines = 1.0, dropouts = 1.0, blur = 1.0):
        B, H, W, C = image.shape
        out = image.clone()
        chroma_strength = int(2 * chroma_shift * random.uniform(-1, 1))
        if chroma_strength != 0:
            for c in [1, 2]:
                shifted = torch.roll(out[..., c], shifts=chroma_strength, dims=2)
                out[..., c] = shifted
        if jitter > 0:
            lines = torch.arange(H, device=image.device).view(1, H, 1, 1).expand(B, H, W, C)
            phase = random.random() * 10
            wobble = (torch.sin(lines.float() * 0.2 + phase) * (2 * jitter)).long()
            for b in range(B):
                for y in range(H):
                    shift = int(wobble[b, y, 0, 0].item())
                    out[b, y] = torch.roll(out[b, y], shifts=shift, dims=0)
        if head_noise > 0 and random.random() < 0.8:
            band_height = int(H * 0.04 * head_noise)
            noise = torch.rand(B, band_height, W, C, device=image.device) * 0.2
            out[:, -band_height:] = noise + out[:, -band_height:] * 0.8
        if scanlines > 0:
            frequency = 2
            intensity = 0.15 * scanlines
            mod_mask = torch.arange(H, device=image.device).float()
            mod_mask = torch.sin((mod_mask / frequency) * 3.1415) ** 2
            mod_mask = 1.0 - (mod_mask * intensity)
            mod_mask = mod_mask.view(1, H, 1, 1)
            out = out * mod_mask
        num_drops = int(3 * dropouts)
        for _ in range(num_drops):
            if random.random() < 0.7:
                y = random.randint(0, H - 2)
                height = random.randint(1, 2)
                drop = torch.rand(B, height, W, C, device=image.device)
                out[:, y:y+height] = drop * 0.6 + out[:, y:y+height] * 0.4
        k = int(3 * blur)
        if k > 0:
            pad = k // 2
            out = F.avg_pool2d(out.permute(0, 3, 1, 2), kernel_size=k, stride=1, padding=pad)
            out = out.permute(0, 2, 3, 1)
        return (torch.clamp(out, 0.0, 1.0),)

class Watermark:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "alias": ("STRING", {"default": ""}),
                "use_lsb": ("BOOLEAN", {"default": True}),
                "repeat_lsb": ("BOOLEAN", {"default": True}),
                "use_visual_watermark": ("BOOLEAN", {"default": True}),
                "visual_watermark_location": (
            [   
                'top left',
                'top right',
                'bottom left',
                'bottom right',
                'all corners',
                'tiled',
            ], {
                "default": 'tiled'
            }),
                "rotate_text": ("BOOLEAN", {"default": True}),
                "rotation_angle": ("INT", {"default": 45, "min": -360, "max": 360, "step": 1}),
                "text_spacing_x": ("INT", {"default": 1, "min": 1, "max": 40, "step": 1}),
                "text_spacing_y": ("INT", {"default": 4, "min": 1, "max": 40, "step": 1}),
                "text_scale": ("INT", {"default": 20, "min": 4, "max": 72, "step": 1}),
                "text_red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "text_green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "text_blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_watermark"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying digital watermarks on an image. It has both least significant bit watermarking and proper text watermarking modes.

alias = text to use for the watermark, can be just an alias or whatever you want, could be the entire prompt for the image
use_lsb = put the alias into the least significant bit of the pixels
repeat_lsb = keep repeating the alias throughout the entire image pixel data
use_visual_watermark = use text watermark visually
visual_watermark_location = where to put the text, multiple locations
rotate_text = rotate text or not
rotation_angle = degree to rotate text to (0 = no rotation)
text_spacing_x and text_spacing_y = for the tile text mode
text_scale = text scale
text_red, text_green, text_blue = text color in 0-255 range
alpha = text alpha

"""

    def func_watermark(self, image: torch.Tensor, alias: str, use_lsb: bool = True, repeat_lsb: bool = True, use_visual_watermark: bool = True, visual_watermark_location: str = 'top left', rotate_text: bool = True, rotation_angle: int = 45, text_spacing_x: int = 1, text_spacing_y: int = 4, text_scale: int = 20, text_red: int = 255, text_green: int = 255, text_blue: int = 255, alpha: float = 0.25):
        assert image.dtype == torch.float32 and image.ndim == 4 and image.shape[-1] == 3, \
            "Expected image of shape [B, H, W, C] with C=3 and dtype float32"
        B, H, W, C = image.shape
        watermarked = []
        font_size = int(min(W, H) / text_scale)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        binary_alias = ''.join(format(ord(c), '08b') for c in alias)
        for b in range(B):
            img = image[b]
            w_img = img.clone()
            if use_visual_watermark:
                canvas = Image.new("RGB", (W, H), (0, 0, 0))
                draw = ImageDraw.Draw(canvas)
                text_bbox = font.getbbox(alias)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_size = (text_width, text_height)
                padding = max(text_size)
                padded_size = (text_size[0] + padding, text_size[1] + padding)
                if visual_watermark_location == 'tiled':
                    for y in range((-padded_size[1]), H, text_size[1] * text_spacing_y):
                        for x in range((-padded_size[0]), W, text_size[0] * text_spacing_x):
                            if rotate_text:
                                txt_img = Image.new("RGB", padded_size, (0, 0, 0))
                                d = ImageDraw.Draw(txt_img)
                                d.text((0, 0), alias, font=font, fill=(text_red, text_green, text_blue))
                                txt_img = txt_img.rotate(rotation_angle, expand=1)
                                mask = txt_img.convert("L").point(lambda p: 255 if p > 0 else 0)
                                canvas.paste(txt_img, (x, y), mask)
                            else:
                                draw.text((x, y), alias, font=font, fill=(text_red, text_green, text_blue))
                elif visual_watermark_location == 'top left':
                    if rotate_text:
                        padding = 2
                        padded_width = text_width + padding * 2
                        padded_height = text_height + padding * 2
                        txt_img = Image.new("L", (padded_width, padded_height), 0)
                        d = ImageDraw.Draw(txt_img)
                        d.text((padding - text_bbox[0], padding - text_bbox[1]), alias, font=font, fill=255)
                        txt_img_rotated = txt_img.rotate(rotation_angle, expand=True)
                        rotated_bbox = txt_img_rotated.getbbox()
                        cropped_txt = txt_img_rotated.crop(rotated_bbox)
                        canvas.paste(Image.merge("RGB", [cropped_txt] * 3), (2, 2), cropped_txt)
                    else:
                        draw.text((2, 2), alias, font=font, fill=(255, 255, 255))
                elif visual_watermark_location == 'top right':
                    if rotate_text:
                        padding = 2
                        padded_width = text_width + padding * 2
                        padded_height = text_height + padding * 2
                        txt_img = Image.new("L", (padded_width, padded_height), 0)
                        d = ImageDraw.Draw(txt_img)
                        d.text((padding - text_bbox[0], padding - text_bbox[1]), alias, font=font, fill=255)
                        txt_img_rotated = txt_img.rotate(rotation_angle, expand=True)
                        rotated_bbox = txt_img_rotated.getbbox()
                        cropped_txt = txt_img_rotated.crop(rotated_bbox)
                        x_position = canvas.width - cropped_txt.width - padding
                        canvas.paste(Image.merge("RGB", [cropped_txt] * 3), (x_position, padding), cropped_txt)
                    else:
                        x_position = canvas.width - text_width - 2
                        draw.text((x_position, 2), alias, font=font, fill=(255, 255, 255))
                elif visual_watermark_location == 'bottom left':
                    if rotate_text:
                        padding = 2
                        padded_width = text_width + padding * 2
                        padded_height = text_height + padding * 2
                        txt_img = Image.new("L", (padded_width, padded_height), 0)
                        d = ImageDraw.Draw(txt_img)
                        d.text((padding - text_bbox[0], padding - text_bbox[1]), alias, font=font, fill=255)
                        txt_img_rotated = txt_img.rotate(rotation_angle, expand=True)
                        rotated_bbox = txt_img_rotated.getbbox()
                        cropped_txt = txt_img_rotated.crop(rotated_bbox)
                        y_position = canvas.height - cropped_txt.height - padding
                        canvas.paste(Image.merge("RGB", [cropped_txt] * 3), (padding, y_position), cropped_txt)
                    else:
                        x_position = 2
                        y_position = canvas.height - text_height - 2
                        draw.text((x_position, y_position), alias, font=font, fill=(255, 255, 255))
                elif visual_watermark_location == 'bottom right':
                    if rotate_text:
                        padding = 2
                        padded_width = text_width + padding * 2
                        padded_height = text_height + padding * 2
                        txt_img = Image.new("L", (padded_width, padded_height), 0)
                        d = ImageDraw.Draw(txt_img)
                        d.text((padding - text_bbox[0], padding - text_bbox[1]), alias, font=font, fill=255)
                        txt_img_rotated = txt_img.rotate(rotation_angle, expand=True)
                        rotated_bbox = txt_img_rotated.getbbox()
                        cropped_txt = txt_img_rotated.crop(rotated_bbox)
                        x_position = canvas.width - cropped_txt.width - padding
                        y_position = canvas.height - cropped_txt.height - padding
                        canvas.paste(Image.merge("RGB", [cropped_txt] * 3), (x_position, y_position), cropped_txt)
                    else:
                        x_position = canvas.width - text_width - 2
                        y_position = canvas.height - text_height - 2
                        draw.text((x_position, y_position), alias, font=font, fill=(255, 255, 255))
                elif visual_watermark_location == 'all corners':
                    padding = 2
                    if rotate_text:
                        padded_width = text_width + padding * 2
                        padded_height = text_height + padding * 2
                        txt_img = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
                        d = ImageDraw.Draw(txt_img)
                        d.text((padding - text_bbox[0], padding - text_bbox[1]), alias, font=font, fill=(text_red, text_green, text_blue))
                        txt_img_rotated = txt_img.rotate(rotation_angle, expand=True)
                        rotated_bbox = txt_img_rotated.getbbox()
                        cropped_txt = txt_img_rotated.crop(rotated_bbox)
                        mask = cropped_txt.convert("L").point(lambda p: 255 if p > 0 else 0)
                        positions = [
                            (padding, padding),
                            (canvas.width - cropped_txt.width - padding, padding),
                            (padding, canvas.height - cropped_txt.height - padding),
                            (canvas.width - cropped_txt.width - padding, canvas.height - cropped_txt.height - padding)
                        ]
                        for pos in positions:
                            canvas.paste(cropped_txt, pos, mask)
                    else:
                        positions = [
                            (padding, padding),
                            (canvas.width - text_width - padding, padding),
                            (padding, canvas.height - text_height - padding),
                            (canvas.width - text_width - padding, canvas.height - text_height - padding)
                        ]
                        for pos in positions:
                            draw.text(pos, alias, font=font, fill=(text_red, text_green, text_blue))
                overlay = TF.to_tensor(canvas).permute(1, 2, 0)
                alpha = alpha
                screen_blend = 1 - (1 - w_img) * (1 - overlay)
                w_img = (1 - alpha) * w_img + alpha * screen_blend
                w_img = torch.clamp(w_img, 0.0, 1.0)
    
            if use_lsb:
                num_pixels = H * W * C
                repeats = math.ceil(num_pixels / len(binary_alias))
                binary_stream = (binary_alias * repeats)[:num_pixels] if repeat_lsb else binary_alias[:num_pixels]
                flat = (w_img * 255).to(torch.uint8).reshape(-1)
                bits = torch.tensor([int(b) for b in binary_stream], dtype=torch.uint8, device=flat.device)
                flat[:bits.numel()] = (flat[:bits.numel()] & 0xFE) | bits
                w_img = flat.reshape(H, W, C).to(torch.float32) / 255.0
            watermarked.append(w_img)
        return (torch.stack(watermarked, dim=0),)
        
class Get_Watermark:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "alias_length": ("INT", {"default": 1, "min": 1, "max": 400, "step": 1}),
                "repeat_lsb": ("BOOLEAN", {"default": True}),
                "print_to_console": ("BOOLEAN", {"default": True}),
            }
        }
        return inputs
    RETURN_TYPES = ("TEXT",)
    OUTPUT_NODE = True
    RETURN_NAMES = ("text",)
    FUNCTION = "func_get_watermark"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for getting the lsb watermark on an image if there is one.

"""

    def func_get_watermark(self, image: torch.Tensor, alias_length: int = 1, repeat_lsb: bool = False, print_to_console: bool = False):
        assert image.dtype == torch.float32 and image.ndim == 4 and image.shape[-1] == 3, \
            "Expected image of shape [B, H, W, C] with C=3 and dtype float32"
        B, H, W, C = image.shape
        aliases = []
        for b in range(B):
            img = image[b]
            flat = (img * 255).to(torch.uint8).flatten()
            if repeat_lsb:
                bits = (flat & 1).tolist()
                bitstream = ''.join(str(bit) for bit in bits)
                chars = [
                    chr(int(bitstream[i:i+8], 2))
                    for i in range(0, len(bitstream) - 8 + 1, 8)
                ]
                alias = ''.join(chars)
            else:
                bits = (flat[:alias_length * 8] & 1).tolist()
                bitstream = ''.join(str(bit) for bit in bits)
                chars = [
                    chr(int(bitstream[i:i+8], 2))
                    for i in range(0, len(bitstream), 8)
                ]
                alias = ''.join(chars)
            alias = alias.split('\x00')[0]
            aliases.append(alias)
            if print_to_console:
                print(''.join(aliases))
        lsb = ''.join(aliases)
        return (lsb,)

class Multi_Scale_Contrast:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "sharpness_factor": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enhancement_strength": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.05}),
                "brightness_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "effect_mode": (
            [   
                '0',
                '1',
                '2',
                '3',
                '4',
            ], {
                "default": '0'
            }),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_enhance_contrast_multi_scale"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying sharpening and contrast adjustments on an image, similar to the reshade effect.

"""

    def func_enhance_contrast_multi_scale(self, image: torch.Tensor, sharpness_factor: float = 0.7, enhancement_strength: float = 0.5, brightness_boost: float = 0.0, effect_mode: str = "2", diagnostics: int = 0):
        B, H, W, C = image.shape
        dev = image.device
        def compute_luma(x):
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=dev)
            return (x * weights).sum(dim=-1, keepdim=True)
        def shrink(x, stride):
            return F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=stride, stride=stride).permute(0, 2, 3, 1)
        def expand(x, target_size):
            return F.interpolate(x.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        def tone_map(x):
            return x / (x + 1.0)
        def tone_unmap(x):
            return -x / (x - 1.0 + 1e-5)
        base_luma = compute_luma(image)
        blurred = shrink(image, 2)
        blurred = expand(blurred, (H, W))
        untoned = tone_unmap(image)
        if effect_mode == "0":
            enhancement = 3.0 * enhancement_strength * untoned * (
                torch.sqrt(blurred + 1e-5) - torch.sqrt(0.5 * base_luma + 0.5 * blurred + 1e-5)
            )
            untoned = untoned - enhancement
        elif effect_mode == "1":
            untoned = untoned * (1.0 - enhancement_strength * blurred)
        elif effect_mode == "2":
            gain = (1.5 + (1.0 - sharpness_factor) ** 2.0) * blurred
            untoned = untoned * (1.0 + 0.9 * enhancement_strength * gain)
        elif effect_mode == "3":
            difference = torch.pow(1.0 - blurred, 0.5)
            untoned = untoned + difference * enhancement_strength * (untoned - blurred)
        elif effect_mode == "4":
            edge_detail = blurred * (0.5 - torch.pow(torch.abs(blurred - base_luma), 0.5) * torch.sign(blurred - base_luma))
            tm = tone_map(untoned)
            sharp_detail = tm * (edge_detail / (blurred + 1e-5))
            screened = 1.0 - (1.0 - tm) * blurred
            final_blend = torch.lerp(sharp_detail, screened, tm)
            blend_amount = torch.clamp(torch.tensor(0.8 * enhancement_strength, device=dev), -0.8, 0.8)
            untoned = torch.lerp(tm, final_blend, blend_amount)
        final_image = tone_map(untoned)
        brightness_adjust = 1.0 + brightness_boost * 0.5 * base_luma ** 2.0
        final_image = final_image ** brightness_adjust
        if diagnostics == 1:
            final_image = torch.sqrt(2.0 * torch.abs(final_image - enhancement_strength * tone_map(untoned)))
        elif diagnostics == 2:
            final_image = torch.sqrt(torch.sum((enhancement_strength * blurred - final_image) ** 2, dim=-1, keepdim=True))
        return (torch.clamp(final_image, 0.0, 1.0),)

class Contrast_Adaptive_Sharpening:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sharpening": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_cas"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying amd's CAS to an image.

"""

    def func_cas(self, image: torch.Tensor, contrast: float = 0.0, sharpening: float = 1.0):
        B, H, W, C = image.shape
        assert C == 3, "Input must be RGB"
        pad = F.pad(image.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='reflect')
        def sample(offset_y, offset_x):
            return pad[:, :, 1 + offset_y : 1 + offset_y + H, 1 + offset_x : 1 + offset_x + W]
        a = sample(-1, -1).permute(0, 2, 3, 1)
        b = sample(-1,  0).permute(0, 2, 3, 1)
        c = sample(-1,  1).permute(0, 2, 3, 1)
        d = sample( 0, -1).permute(0, 2, 3, 1)
        e = sample( 0,  0).permute(0, 2, 3, 1)
        f = sample( 0,  1).permute(0, 2, 3, 1)
        g = sample( 1, -1).permute(0, 2, 3, 1)
        h = sample( 1,  0).permute(0, 2, 3, 1)
        i = sample( 1,  1).permute(0, 2, 3, 1)
        mnRGB = torch.min(torch.min(torch.min(d, e), torch.min(f, b)), h)
        mnRGB2 = torch.min(mnRGB, torch.min(torch.min(a, c), torch.min(g, i)))
        mnRGB += mnRGB2
        mxRGB = torch.max(torch.max(torch.max(d, e), torch.max(f, b)), h)
        mxRGB2 = torch.max(mxRGB, torch.max(torch.max(a, c), torch.max(g, i)))
        mxRGB += mxRGB2
        rcpMRGB = 1.0 / (mxRGB + 1e-5)
        ampRGB = torch.clamp(torch.min(mnRGB, 2.0 - mxRGB) * rcpMRGB, 0, 1)
        ampRGB = torch.rsqrt(ampRGB + 1e-5)
        peak = -3.0 * contrast + 8.0
        wRGB = -1.0 / (ampRGB * peak + 1e-5)
        rcpWeightRGB = 1.0 / (4.0 * wRGB + 1.0)
        window = b + d + f + h
        outColor = torch.clamp((window * wRGB + e) * rcpWeightRGB, 0, 1)
        return (torch.lerp(e, outColor, sharpening),)

class VHS_Chroma_Smear:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "smear_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
                "time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005}),
                "uniform": ("BOOLEAN", {"default": True}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_vhs_chroma_smear"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying VHS chroma smearing on an image.

"""

    def func_vhs_chroma_smear(self, image: torch.Tensor, smear_strength: float = 0.5, time: float = 0.0, uniform: bool = True):
        B, H, W, C = image.shape
        device = image.device
        smear_strength = smear_strength * 100
        def rgb_to_yiq(rgb):
            r, g, b = rgb.unbind(-1)
            y = 0.2989 * r + 0.5959 * g + 0.2115 * b
            i = 0.5870 * r - 0.2744 * g - 0.5229 * b
            q = 0.1140 * r - 0.3216 * g + 0.3114 * b
            return torch.stack([y, i, q], dim=-1)
        def yiq_to_rgb(yiq):
            y, i, q = yiq.unbind(-1)
            r = y + i + q
            g = 0.956 * y - 0.2720 * i - 1.1060 * q
            b = 0.6210 * y - 0.6474 * i + 1.7046 * q
            return torch.stack([r, g, b], dim=-1)
        def circle_kernel(n_points, start_angle):
            θ = 2 * math.pi * (torch.arange(n_points, device=device) + start_angle) / n_points
            return torch.stack([-(0.3 + θ), torch.cos(θ)], dim=-1)
        uv_grid = F.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1),
            size=[B, C, H, W],
            align_corners=False
        )
        def blur_pass(offset_scale):
            k = circle_kernel(14, start_angle=2.0 / 14.0).to(device)
            outputs = []
            for b in range(B):
                if isinstance(offset_scale, torch.Tensor):
                    os_b = offset_scale[b]
                    os_b = os_b.view(1, H, W, 1)
                    offsets = (k.view(14, 1, 1, 2) * os_b)
                else:
                    offsets = k.view(14, 1, 1, 2) * offset_scale
                grid_b = (uv_grid[b].unsqueeze(0).expand(14, -1, -1, -1) + offsets).clamp(-1, 1)
                inp_b   = image[b : b+1].permute(0, 3, 1, 2)
                inp_rep = inp_b.repeat(14, 1, 1, 1)
                sampled = F.grid_sample(inp_rep, grid_b, mode='bilinear', padding_mode='reflection', align_corners=False)  
                mean_b   = sampled.mean(0)                          
                outputs.append(mean_b.permute(1, 2, 0))             
            return torch.stack(outputs, dim=0)
        t = torch.tensor(time, device=device)
        d = 0.051 + torch.abs(torch.sin(t / 4.0))
        base_offset = max(1e-4, 0.002 * d.item()) * smear_strength
        uv01 = (uv_grid[..., 0] + 1.0) * 0.5
        if uniform:
            offset_map_y = base_offset
        else:
            offset_map_y = base_offset + base_offset * uv01
        blurred_y = rgb_to_yiq(blur_pass(offset_map_y))[..., 0:1]
        blurred_i = rgb_to_yiq(blur_pass(base_offset * 6.0))[..., 1:2]
        blurred_q = rgb_to_yiq(blur_pass(base_offset * 2.5))[..., 2:3]
        final = yiq_to_rgb(torch.cat([blurred_y, blurred_i, blurred_q], dim=-1))
        return (final.clamp(0.0, 1.0),)

class NTSC_Filter:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.005}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_ntsc_filter"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying the ntsc filter on an image.

"""

    def func_ntsc_filter(self, image: torch.Tensor, opacity: float):
        x = image.permute(0, 3, 1, 2).float()
        B, C, H, W = x.shape
        device = x.device
        YAccum = torch.zeros((B, H, W, 4), device=device, dtype=torch.float32)
        IAccum = torch.zeros((B, H, W, 4), device=device, dtype=torch.float32)
        QAccum = torch.zeros((B, H, W, 4), device=device, dtype=torch.float32)
        CCFrequency = 3.59754545
        YFrequency  = 6.0
        IFrequency  = 1.2
        QFrequency  = 0.6
        NotchHalfWidth = 2.0
        ScanTime = 52.6
        MinC = -1.1183
        CRange = 3.2366
        QuadXSize = float(W) * 4.0
        TimePerSample = ScanTime / QuadXSize
        Fc_y1 = (CCFrequency - NotchHalfWidth) * TimePerSample
        Fc_y2 = (CCFrequency + NotchHalfWidth) * TimePerSample
        Fc_y3 = YFrequency * TimePerSample
        Fc_i  = IFrequency * TimePerSample
        Fc_q  = QFrequency * TimePerSample
        Pi2 = 2.0 * torch.pi
        Pi2Length = Pi2 / 82.0
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid_base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        pixel_shift = 2.0 / float(W - 1)
        offsets = [0.25 * pixel_shift, 0.50 * pixel_shift, 0.75 * pixel_shift]
        i_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
        j_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        for n in range(-41, 42, 4):
            offset_norm = n * 0.25 * pixel_shift
            grid_n = grid_base.clone()
            grid_n[..., 0] = grid_n[..., 0] + offset_norm
            Texel0 = F.grid_sample(x, grid_n, mode='bilinear', padding_mode='border', align_corners=True)
            grid_n1 = grid_n.clone()
            grid_n2 = grid_n.clone()
            grid_n3 = grid_n.clone()
            grid_n1[..., 0] += offsets[0]
            grid_n2[..., 0] += offsets[1]
            grid_n3[..., 0] += offsets[2]
            Texel1 = F.grid_sample(x, grid_n1, mode='bilinear', padding_mode='border', align_corners=True)
            Texel2 = F.grid_sample(x, grid_n2, mode='bilinear', padding_mode='border', align_corners=True)
            Texel3 = F.grid_sample(x, grid_n3, mode='bilinear', padding_mode='border', align_corners=True)
            R0, G0, B0 = Texel0[:,0:1], Texel0[:,1:2], Texel0[:,2:3]
            R1, G1, B1 = Texel1[:,0:1], Texel1[:,1:2], Texel1[:,2:3]
            R2, G2, B2 = Texel2[:,0:1], Texel2[:,1:2], Texel2[:,2:3]
            R3, G3, B3 = Texel3[:,0:1], Texel3[:,1:2], Texel3[:,2:3]
            Y0 = 0.299 * R0 + 0.587 * G0 + 0.114 * B0
            Y1 = 0.299 * R1 + 0.587 * G1 + 0.114 * B1
            Y2 = 0.299 * R2 + 0.587 * G2 + 0.114 * B2
            Y3 = 0.299 * R3 + 0.587 * G3 + 0.114 * B3
            I0 = 0.595716 * R0 - 0.274453 * G0 - 0.321263 * B0
            I1 = 0.595716 * R1 - 0.274453 * G1 - 0.321263 * B1
            I2 = 0.595716 * R2 - 0.274453 * G2 - 0.321263 * B2
            I3 = 0.595716 * R3 - 0.274453 * G3 - 0.321263 * B3
            Q0 = 0.211456 * R0 - 0.522591 * G0 + 0.311135 * B0
            Q1 = 0.211456 * R1 - 0.522591 * G1 + 0.311135 * B1
            Q2 = 0.211456 * R2 - 0.522591 * G2 + 0.311135 * B2
            Q3 = 0.211456 * R3 - 0.522591 * G3 + 0.311135 * B3
            Yv = torch.cat((Y0, Y1, Y2, Y3), dim=1).permute(0,2,3,1)
            Iv = torch.cat((I0, I1, I2, I3), dim=1).permute(0,2,3,1)
            Qv = torch.cat((Q0, Q1, Q2, Q3), dim=1).permute(0,2,3,1)
            i_coords = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
            j_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
            i_exp = i_coords.unsqueeze(-1)
            j_exp = j_coords.unsqueeze(-1)
            k_off = torch.tensor([0.0, 0.25, 0.50, 0.75], device=device).view(1, 1, 4)
            T_base = i_exp * float(W) + 0.5    \
                    + (j_exp + k_off)
            Wconst = 2 * torch.pi * CCFrequency * ScanTime
            T_total = T_base * Wconst
            cosT = torch.cos(T_total).unsqueeze(0).expand(B, H, W, 4)
            sinT = torch.sin(T_total).unsqueeze(0).expand(B, H, W, 4)
            Encoded = Yv + Iv * cosT + Qv * sinT
            Cnorm = (Encoded + (-MinC)) / CRange
            Cval = Cnorm * CRange + MinC
            n4 = torch.tensor([n, n+1, n+2, n+3], device=device, dtype=torch.float32) + 1e-5
            Sinc = lambda x: torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x)/x)
            SincY1 = Sinc(Pi2 * Fc_y1 * n4)
            SincY2 = Sinc(Pi2 * Fc_y2 * n4)
            SincY3 = Sinc(Pi2 * Fc_y3 * n4)
            SincI  = Sinc(Pi2 * Fc_i  * n4)
            SincQ  = Sinc(Pi2 * Fc_q  * n4)
            IdealY = (2.0 * Fc_y1 * SincY1 - 2.0 * Fc_y2 * SincY2) + 2.0 * Fc_y3 * SincY3
            IdealI = 2.0 * Fc_i  * SincI
            IdealQ = 2.0 * Fc_q  * SincQ
            Window = 0.54 + 0.46 * torch.cos(Pi2Length * n4)
            FilterY = Window * IdealY
            FilterI = Window * IdealI
            FilterQ = Window * IdealQ
            Term = (j_coords.unsqueeze(-1)/W + (i_coords.unsqueeze(-1)/H)*float(W) + 0.5 + n4*0.25/float(W))
            phase = Wconst * Term
            cosPhase = torch.cos(phase).unsqueeze(0).expand(B, H, W, 4)
            sinPhase = torch.sin(phase).unsqueeze(0).expand(B, H, W, 4)
            FY = FilterY.view(1,1,1,4)
            FI = FilterI.view(1,1,1,4)
            FQ = FilterQ.view(1,1,1,4)
            YAccum += Cval * FY
            IAccum += Cval * cosPhase * FI
            QAccum += Cval * sinPhase * FQ
        Y = YAccum.sum(dim=-1)
        I = IAccum.sum(dim=-1) * 2.0
        Q = QAccum.sum(dim=-1) * 2.0
        R = Y + 0.956 * I + 0.621 * Q
        G = Y - 0.272 * I - 0.647 * Q
        Bc = Y - 1.106 * I + 1.703 * Q
        Out = torch.stack((R, G, Bc), dim=-1)
        if C == 4:
            alpha = input_tensor[..., 3:4]
            Out = torch.cat((Out, alpha), dim=-1)
        opacity = 0.75
        if opacity != 1:
            blended = (1 - opacity) * image + opacity * Out
        else:
            blended = Out
        return (blended,)
    
class Pixel_Art:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "scale": ("INT", {"default": 2, "min": 1, "max": 128, "step": 1}),
                "num_colors": ("INT", {"default": 4, "min": 1, "max": 256, "step": 1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 3.0, "step": 0.05}),
                "dither": (
            [   
                'none',
                'floyd',
                'noise',
                'ordered',
                'pattern',
            ], {
                "default": 'none'
            }),
                "matrix_size": (
            [   
                '2',
                '4',
                '8',
                '16',
                '32',
            ], {
                "default": '4'
            }),
                "palette_mode": (
            [   
                'none',
                'adaptive',
                'median_cut',
                'perceptual',
                'uniform',
                'consolidate',
            ], {
                "default": 'none'
            }),
                "dither_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "matrix_expand_negative": ("BOOLEAN", {"default": True}),
                "floyd_dither_serpentine": ("BOOLEAN", {"default": False}),
                "floyd_dither_threshold_jitter": ("BOOLEAN", {"default": False}),
                "palette_filepath": ("STRING", {"default": ""}),
            },
            "optional":
            {
                "palette": ("IMAGE",),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_pixel_art"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying pixel art effects to an image. This is very customizable, with multiple controls like number of colors to use, dither mode, dither matrix size, palette mode, matrix threshold and expanding to negative values too. If you provide a palette it expects each pixel to be a unique color so say 4x1 or 2x2 are fine.

scale = what to divide the image size by, higher values means bigger pixels
num_colors = how many colors the palette should have, some palette modes may not use exactly the number if its really high
saturation = pre add more saturation to the image before making the palette
dither = dither mode for quantization
matrix_size = size of the bayer matrix for dithering
palette_mode = mode for creating the palette of the image
dither_strength = how much dither to apply
matrix_expand_negative = expand the matrix_threshold into the negative space, good for higher num_colors
floyd_dither_serpentine = serpentine the dither or not
floyd_dither_threshold_jitter = jitter the dither or not
palette_filepath = path to any .ase, .aco or .act swatch file to use as the palette

"""

    def func_pixel_art(self, image, scale: float = 0.25, num_colors: int = 8, saturation: float = 1.5, dither: str = "none", matrix_size: str = '4', palette: torch.Tensor = None, palette_mode: str = "adaptive", dither_strength: float = 1.0, matrix_expand_negative: bool = True, floyd_dither_serpentine: bool = False, floyd_dither_threshold_jitter: bool = False, palette_filepath: str = ""):
        matrix_size = int(matrix_size)
        matrix_threshold = dither_strength
        def _cmyk_to_rgb(c, m, y, k):
            C = 1 - c
            M = 1 - m
            Y = 1 - y
            K = 1 - k
            R = C * K
            G = M * K
            B = Y * K
            return R, G, B
        def _lab_to_rgb(L, a, b):
            Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
            def finv(t):
                delta = 6/29
                return torch.where(
                    t > delta,
                    t**3,
                    3 * delta**2 * (t - 4/29)
                )
            fy = (L + 16.0) / 116.0
            fx = fy + (a / 500.0)
            fz = fy - (b / 200.0)
            X = Xn * finv(fx)
            Y = Yn * finv(fy)
            Z = Zn * finv(fz)
            M = torch.tensor([
                [ 3.2406, -1.5372, -0.4986],
                [-0.9689,  1.8758,  0.0415],
                [ 0.0557, -0.2040,  1.0570],
            ])
            rgb_lin = torch.matmul(M, torch.stack([X, Y, Z], dim=-1)[...,None])[...,0]
            def compand(u):
                a = 0.055
                return torch.where(
                    u <= 0.0031308,
                    12.92 * u,
                    (1 + a) * torch.pow(u, 1/2.4) - a
                )
            rgb = compand(rgb_lin)
            return torch.clamp(rgb, 0.0, 1.0)
        def load_aco_swatch(filepath: str) -> torch.Tensor:
            with open(filepath, 'rb') as f:
                data = f.read()
            offset = 0
            if len(data) < 4:
                raise ValueError("Not a valid .aco (too small)")
            version1, count1 = struct.unpack_from('>HH', data, offset)
            offset += 4
            if version1 != 1:
                raise ValueError(f"Expected v1 header==1, got {version1}")
            colors = []
            for i in range(count1):
                if offset + 10 > len(data):
                    raise ValueError(f"Unexpected EOF reading v1 entry #{i}")
                space, d1, d2, d3, d4 = struct.unpack_from('>HHHHH', data, offset)
                offset += 10
                if space == 0:
                    r, g, b = d1 >> 8, d2 >> 8, d3 >> 8
                else:
                    r, g, b = d1 >> 8, d2 >> 8, d3 >> 8
                    warnings.warn(f"v1 entry #{i}: unsupported space {space}, using raw bytes")
                colors.append((r, g, b))
            try:
                if offset + 4 > len(data):
                    raise ValueError("No room for v2 header")
                version2, count2 = struct.unpack_from('>HH', data, offset)
                offset += 4
                if version2 != 2 or count2 <= 0:
                    raise ValueError("No valid v2 block")
                colors_v2 = []
                for i in range(count2):
                    if offset + 10 > len(data):
                        raise ValueError(f"Unexpected EOF reading v2 entry #{i}")
                    space, d1, d2, d3, d4 = struct.unpack_from('>HHHHH', data, offset)
                    offset += 10
                    name_len = struct.unpack_from('>H', data, offset)[0]
                    offset += 2 + name_len * 2
                    if space == 0:
                        r, g, b = d1 >> 8, d2 >> 8, d3 >> 8
                    else:
                        r, g, b = d1 >> 8, d2 >> 8, d3 >> 8
                        warnings.warn(f"v2 entry #{i}: unsupported space {space}, using raw bytes")
                    colors_v2.append((r, g, b))
                colors = colors_v2
            except Exception as e:
                warnings.warn(f"Skipping v2 block (reason: {e}) and using v1 colors")
            if not colors:
                raise ValueError("No parsable colors found in .aco file")
            palette = torch.tensor(colors, dtype=torch.uint8)
            palette = palette.float().div(255.0)
            return palette.unsqueeze(0).unsqueeze(0)
        def load_ase_swatch(filepath: str) -> torch.Tensor:
            with open(filepath, 'rb') as f:
                data = f.read()
            offset = 0
            sig, ver_major, ver_minor, n_blocks = struct.unpack_from('>4sHHI', data, offset)
            offset += 4 + 2 + 2 + 4
            if sig != b'ASEF':
                raise ValueError("Not a valid ASE file (missing ASEF signature)")
            colors = []
            for blk in range(n_blocks):
                if offset + 6 > len(data):
                    warnings.warn(f"Unexpected EOF before block #{blk}, stopping.")
                    break
                blk_type, blk_len = struct.unpack_from('>HI', data, offset)
                offset += 6
                blk_data = data[offset: offset + blk_len]
                offset += blk_len
                if blk_type != 0x0001:
                    continue
                name_len = struct.unpack_from('>H', blk_data, 0)[0]
                pos = 2 + name_len * 2
                model = blk_data[pos:pos+4].decode('ascii')
                pos += 4
                try:
                    if model == 'RGB ':
                        r_f, g_f, b_f = struct.unpack_from('>fff', blk_data, pos)
                        pos += 12
                    elif model == 'GRAY':
                        (g_f,) = struct.unpack_from('>f', blk_data, pos)
                        pos += 4
                        r_f = g_f; g_f = g_f; b_f = g_f
                    elif model == 'CMYK':
                        c_f, m_f, y_f, k_f = struct.unpack_from('>ffff', blk_data, pos)
                        pos += 16
                        r_f, g_f, b_f = _cmyk_to_rgb(c_f, m_f, y_f, k_f)
                    else:
                        warnings.warn(f"Skipping unsupported model '{model}'")
                        continue
                except struct.error:
                    warnings.warn(f"Malformed channel data in block #{blk}, skipping")
                    continue
                r = int(max(0.0, min(1.0, r_f)) * 255)
                g = int(max(0.0, min(1.0, g_f)) * 255)
                b = int(max(0.0, min(1.0, b_f)) * 255)
                colors.append((r, g, b))
            if not colors:
                raise ValueError("No valid colors found in .ase file")
            palette = torch.tensor(colors, dtype=torch.uint8)
            palette = palette.float().div(255.0)
            return palette.unsqueeze(0).unsqueeze(0)
        def load_act_swatch(filepath: str) -> torch.Tensor:
            with open(filepath, 'rb') as f:
                data = f.read()
            if len(data) not in [768, 772]:
                raise ValueError(f"Invalid ACT file length: {len(data)} bytes")
            palette = torch.tensor(list(data[:768]), dtype=torch.uint8).reshape(-1, 3)
            if len(data) == 772:
                num_colors = int.from_bytes(data[768:770], 'big')
                if 0 < num_colors <= 256:
                    palette = palette[:num_colors]
            palette = palette.float() / 255.0
            palette = palette.unsqueeze(0).unsqueeze(0)
            return palette
        if palette_filepath != "" and palette == None:
            palette_filepath = palette_filepath.strip('\'"')
            palette_file_ext = os.path.splitext(palette_filepath)[1].lower()
            if palette_file_ext == '.act':
                palette_file = load_act_swatch(palette_filepath)
            elif palette_file_ext == '.aco':
                palette_file = load_aco_swatch(palette_filepath)
            elif palette_file_ext == '.ase':
                palette_file = load_ase_swatch(palette_filepath)
            palette = palette_file
        def make_bayer_matrix(size: int, device) -> torch.Tensor:
            if size == 2:
                base = torch.tensor([[0, 2], [3, 1]], dtype=torch.float32)
            elif size == 4:
                base = torch.tensor([
                    [ 0,  8,  2, 10],
                    [12,  4, 14,  6],
                    [ 3, 11,  1,  9],
                    [15,  7, 13,  5]
                ], dtype=torch.float32)
            elif size == 8:
                base = torch.tensor([
                    [ 0,32, 8,40, 2,34,10,42],
                    [48,16,56,24,50,18,58,26],
                    [12,44, 4,36,14,46, 6,38],
                    [60,28,52,20,62,30,54,22],
                    [ 3,35,11,43, 1,33, 9,41],
                    [51,19,59,27,49,17,57,25],
                    [15,47, 7,39,13,45, 5,37],
                    [63,31,55,23,61,29,53,21]
                ], dtype=torch.float32)
            elif size == 16:
                base = torch.tensor([
                    [  0,128, 32,160,   8,136, 40,168,  2,130, 34,162, 10,138, 42,170],
                    [192, 64,224, 96, 200, 72,232,104,194, 66,226, 98,202, 74,234,106],
                    [ 48,176, 16,144,  56,184, 24,152, 50,178, 18,146, 58,186, 26,154],
                    [240,112,208, 80, 248,120,216, 88,242,114,210, 82,250,122,218, 90],
                    [ 12,140, 44,172,  4,132,  36,164, 14,142, 46,174,  6,134, 38,166],
                    [204, 76,236,108, 196, 68,228,100,206, 78,238,110,198, 70,230,102],
                    [ 60,188, 28,156,  52,180, 20,148, 62,190, 30,158, 54,182, 22,150],
                    [252,124,220, 92, 244,116,212, 84,254,126,222, 94,246,118,214, 86],
                    [  3,131, 35,163,  11,139, 43,171,  1,129, 33,161, 9, 137, 41,169],
                    [195, 67,227, 99, 203, 75,235,107,193, 65,225, 97,201, 73,233,105],
                    [ 51,179, 19,147,  59,187, 27,155, 49,177, 17,145, 57,185, 25,153],
                    [243,115,211, 83, 251,123,219, 91,241,113,209, 81,249,121,217, 89],
                    [ 15,143, 47,175,   7,135, 39,167, 13,141, 45,173,  5,133, 37,165],
                    [207, 79,239,111, 199, 71,231,103,205, 77,237,109,197, 69,229,101],
                    [ 63,191, 31,159,  55,183, 23,151, 61,189, 29,157, 53,181, 21,149],
                    [255,127,223, 95, 247,119,215, 87,253,125,221, 93,245,117,213, 85],
                ], dtype=torch.float32)
            elif size == 32:
                base = torch.tensor([
                    [   0, 256,  64, 320,  16, 272,  80, 336,   4, 260,  68, 324,  20, 276,  84, 340,   1, 257,  65, 321,  17, 273,  81, 337,   5, 261,  69, 325,  21, 277,  85, 341],
                    [ 384, 128, 448, 192, 400, 144, 464, 208, 388, 132, 452, 196, 404, 148, 468, 212, 385, 129, 449, 193, 401, 145, 465, 209, 389, 133, 453, 197, 405, 149, 469, 213],
                    [  96, 352,  32, 288, 112, 368,  48, 304, 100, 356,  36, 292, 116, 372,  52, 308,  97, 353,  33, 289, 113, 369,  49, 305, 101, 357,  37, 293, 117, 373,  53, 309],
                    [ 480, 224, 416, 160, 496, 240, 432, 176, 484, 228, 420, 164, 500, 244, 436, 180, 481, 225, 417, 161, 497, 241, 433, 177, 485, 229, 421, 165, 501, 245, 437, 181],
                    [  24, 280,  88, 344,   8, 264,  72, 328,  28, 284,  92, 348,  12, 268,  76, 332,  25, 281,  89, 345,   9, 265,  73, 329,  29, 285,  93, 349,  13, 269,  77, 333],
                    [ 408, 152, 472, 216, 392, 136, 456, 200, 412, 156, 476, 220, 396, 140, 460, 204, 409, 153, 473, 217, 393, 137, 457, 201, 413, 157, 477, 221, 397, 141, 461, 205],
                    [ 120, 376,  56, 312, 104, 360,  40, 296, 124, 380,  60, 316, 108, 364,  44, 300, 121, 377,  57, 313, 105, 361,  41, 297, 125, 381,  61, 317, 109, 365,  45, 301],
                    [ 504, 248, 440, 184, 488, 232, 424, 168, 508, 252, 444, 188, 492, 236, 428, 172, 505, 249, 441, 185, 489, 233, 425, 169, 509, 253, 445, 189, 493, 237, 429, 173],
                    [   2, 258,  66, 322,  18, 274,  82, 338,   6, 262,  70, 326,  22, 278,  86, 342,   3, 259,  67, 323,  19, 275,  83, 339,   7, 263,  71, 327,  23, 279,  87, 343],
                    [ 386, 130, 450, 194, 402, 146, 466, 210, 390, 134, 454, 198, 406, 150, 470, 214, 387, 131, 451, 195, 403, 147, 467, 211, 391, 135, 455, 199, 407, 151, 471, 215],
                    [  98, 354,  34, 290, 114, 370,  50, 306, 102, 358,  38, 294, 118, 374,  54, 310,  99, 355,  35, 291, 115, 371,  51, 307, 103, 359,  39, 295, 119, 375,  55, 311],
                    [ 482, 226, 418, 162, 498, 242, 434, 178, 486, 230, 422, 166, 502, 246, 438, 182, 483, 227, 419, 163, 499, 243, 435, 179, 487, 231, 423, 167, 503, 247, 439, 183],
                    [  26, 282,  90, 346,  10, 266,  74, 330,  30, 286,  94, 350,  14, 270,  78, 334,  27, 283,  91, 347,  11, 267,  75, 331,  31, 287,  95, 351,  15, 271,  79, 335],
                    [ 410, 154, 474, 218, 394, 138, 458, 202, 414, 158, 478, 222, 398, 142, 462, 206, 411, 155, 475, 219, 395, 139, 459, 203, 415, 159, 479, 223, 399, 143, 463, 207],
                    [ 122, 378,  58, 314, 106, 362,  42, 298, 126, 382,  62, 318, 110, 366,  46, 302, 123, 379,  59, 315, 107, 363,  43, 299, 127, 383,  63, 319, 111, 367,  47, 303],
                    [ 506, 250, 442, 186, 490, 234, 426, 170, 510, 254, 446, 190, 494, 238, 430, 174, 507, 251, 443, 187, 491, 235, 427, 171, 511, 255, 447, 191, 495, 239, 431, 175],
                    [   1, 257,  65, 321,  17, 273,  81, 337,   5, 261,  69, 325,  21, 277,  85, 341,   0, 256,  64, 320,  16, 272,  80, 336,   4, 260,  68, 324,  20, 276,  84, 340],
                    [ 385, 129, 449, 193, 401, 145, 465, 209, 389, 133, 453, 197, 405, 149, 469, 213, 384, 128, 448, 192, 400, 144, 464, 208, 388, 132, 452, 196, 404, 148, 468, 212],
                    [  97, 353,  33, 289, 113, 369,  49, 305, 101, 357,  37, 293, 117, 373,  53, 309,  96, 352,  32, 288, 112, 368,  48, 304, 100, 356,  36, 292, 116, 372,  52, 308],
                    [ 481, 225, 417, 161, 497, 241, 433, 177, 485, 229, 421, 165, 501, 245, 437, 181, 480, 224, 416, 160, 496, 240, 432, 176, 484, 228, 420, 164, 500, 244, 436, 180],
                    [  25, 281,  89, 345,   9, 265,  73, 329,  29, 285,  93, 349,  13, 269,  77, 333,  24, 280,  88, 344,   8, 264,  72, 328,  28, 284,  92, 348,  12, 268,  76, 332],
                    [ 409, 153, 473, 217, 393, 137, 457, 201, 413, 157, 477, 221, 397, 141, 461, 205, 408, 152, 472, 216, 392, 136, 456, 200, 412, 156, 476, 220, 396, 140, 460, 204],
                    [ 121, 377,  57, 313, 105, 361,  41, 297, 125, 381,  61, 317, 109, 365,  45, 301, 120, 376,  56, 312, 104, 360,  40, 296, 124, 380,  60, 316, 108, 364,  44, 300],
                    [ 505, 249, 441, 185, 489, 233, 425, 169, 509, 253, 445, 189, 493, 237, 429, 173, 504, 248, 440, 184, 488, 232, 424, 168, 508, 252, 444, 188, 492, 236, 428, 172],
                    [   3, 259,  67, 323,  19, 275,  83, 339,   7, 263,  71, 327,  23, 279,  87, 343,   2, 258,  66, 322,  18, 274,  82, 338,   6, 262,  70, 326,  22, 278,  86, 342],
                    [ 387, 131, 451, 195, 403, 147, 467, 211, 391, 135, 455, 199, 407, 151, 471, 215, 386, 130, 450, 194, 402, 146, 466, 210, 390, 134, 454, 198, 406, 150, 470, 214],
                    [  99, 355,  35, 291, 115, 371,  51, 307, 103, 359,  39, 295, 119, 375,  55, 311,  98, 354,  34, 290, 114, 370,  50, 306, 102, 358,  38, 294, 118, 374,  54, 310],
                    [ 483, 227, 419, 163, 499, 243, 435, 179, 487, 231, 423, 167, 503, 247, 439, 183, 482, 226, 418, 162, 498, 242, 434, 178, 486, 230, 422, 166, 502, 246, 438, 182],
                    [  27, 283,  91, 347,  11, 267,  75, 331,  31, 287,  95, 351,  15, 271,  79, 335,  26, 282,  90, 346,  10, 266,  74, 330,  30, 286,  94, 350,  14, 270,  78, 334],
                    [ 411, 155, 475, 219, 395, 139, 459, 203, 415, 159, 479, 223, 399, 143, 463, 207, 410, 154, 474, 218, 394, 138, 458, 202, 414, 158, 478, 222, 398, 142, 462, 206],
                    [ 123, 379,  59, 315, 107, 363,  43, 299, 127, 383,  63, 319, 111, 367,  47, 303, 122, 378,  58, 314, 106, 362,  42, 298, 126, 382,  62, 318, 110, 366,  46, 302],
                    [ 507, 251, 443, 187, 491, 235, 427, 171, 511, 255, 447, 191, 495, 239, 431, 175, 506, 250, 442, 186, 490, 234, 426, 170, 510, 254, 446, 190, 494, 238, 430, 174],
                ], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported Bayer size {size}")
            return (base / (size * size)).to(device)
        def ordered_dither(img: torch.Tensor, levels: int, threshold_map: torch.Tensor) -> torch.Tensor:
            bias = threshold_map / levels
            biased = (img + bias).clamp(0.0, 1.0)
            q = torch.round(biased * (levels - 1)) / (levels - 1)
            return q.clamp(0.0, 1.0)
        def floyd_steinberg_dither(img: torch.Tensor, levels: int, dither_amount: float = 1.0, serpentine: bool = False, threshold_jitter: bool = False) -> torch.Tensor:
            B, C, H, W = img.shape
            out = img.clone()
            for b in range(B):
                for y in range(H):
                    l2r = not serpentine or (y % 2 == 0)
                    x_range = range(W) if l2r else range(W - 1, -1, -1)
                    for x in x_range:
                        old = out[b, :, y, x]
                        if threshold_jitter:
                            noise = (torch.rand_like(old) - 0.5) / (levels - 1)
                        else:
                            noise = 0.0
                        old_n = (old + noise).clamp(0.0, 1.0)
                        new = torch.round(old_n * (levels - 1)) / (levels - 1)
                        out[b, :, y, x] = new
                        err = (old_n - new) * dither_amount
                        if l2r:
                            neighbors = [
                                ( x + 1, y    , 7 / 16),
                                ( x - 1, y + 1, 3 / 16),
                                ( x    , y + 1, 5 / 16),
                                ( x + 1, y + 1, 1 / 16),
                            ]
                        else:
                            neighbors = [
                                ( x - 1, y    , 7 / 16),
                                ( x + 1, y + 1, 3 / 16),
                                ( x    , y + 1, 5 / 16),
                                ( x - 1, y + 1, 1 / 16),
                            ]
                        for nx, ny, w in neighbors:
                            if 0 <= nx < W and 0 <= ny < H:
                                out[b, :, ny, nx] += err * w
            return out.clamp(0.0, 1.0)
        def noise_dither(img: torch.Tensor, levels: int, noise_amount: float = 1.0) -> torch.Tensor:
            out = img.clone().clamp(0.0, 1.0)
            step = 1.0 / (levels - 1)
            jitter = (torch.rand_like(out) - 0.5) * noise_amount * step
            noised = (out + jitter).clamp(0.0, 1.0)
            dithered = torch.round(noised * (levels - 1)) / (levels - 1)
            return dithered.clamp(0.0, 1.0)
        def kmeans_palette(img: torch.Tensor,
                        num_colors: int,
                        iters: int = 10) -> torch.Tensor:
            pixels = img[0].reshape(-1, 3)
            idx = torch.randperm(pixels.size(0), device=pixels.device)[:num_colors]
            centers = pixels[idx].clone()
            for _ in range(iters):
                dists = torch.cdist(pixels, centers)
                labels = dists.argmin(dim=1)
                for k in range(num_colors):
                    pts = pixels[labels == k]
                    if pts.numel():
                        centers[k] = pts.mean(dim=0)
            return centers.clamp(0.0, 1.0)
        def perceptual_palette(img: torch.Tensor, num_colors: int, max_candidates: int = 4096) -> torch.Tensor:
            def rgb_to_hsv(pixels):
                r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
                maxc, _ = pixels.max(dim=1)
                minc, _ = pixels.min(dim=1)
                v = maxc
                delta = maxc - minc + 1e-5
                s = delta / (maxc + 1e-5)
                h = torch.zeros_like(v)
                mask = delta > 0
                r_eq = (maxc == r) & mask
                g_eq = (maxc == g) & mask
                b_eq = (maxc == b) & mask
                h[r_eq] = ((g[r_eq] - b[r_eq]) / delta[r_eq]) % 6
                h[g_eq] = ((b[g_eq] - r[g_eq]) / delta[g_eq]) + 2
                h[b_eq] = ((r[b_eq] - g[b_eq]) / delta[b_eq]) + 4
                h = h / 6.0
                h = h % 1.0
                return torch.stack([h, s, v], dim=1)
            pixels = img[0].reshape(-1, 3)
            if pixels.shape[0] > max_candidates:
                idx = torch.randperm(pixels.shape[0], device=img.device)[:max_candidates]
                pixels = pixels[idx]
            pixels_i = (pixels * 255.0).round().to(torch.int64)
            uniques, counts = torch.unique(pixels_i, return_counts=True, dim=0)
            candidates = uniques.to(torch.float32) / 255.0
            candidates_hsv = rgb_to_hsv(candidates)
            if len(candidates) <= num_colors:
                return candidates[:num_colors]
            idx = counts.argmax()
            palette = [candidates[idx]]
            palette_hsv = [candidates_hsv[idx]]
            mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=img.device)
            mask[idx] = False
            for _ in range(1, num_colors):
                if mask.sum() == 0:
                    break
                current = torch.stack(palette_hsv)
                remaining = candidates_hsv[mask]
                dists = torch.cdist(remaining, current)
                min_dist = dists.min(dim=1).values
                pick_idx = torch.argmax(min_dist)
                pick_global_idx = torch.arange(mask.size(0), device=img.device)[mask][pick_idx]
                palette.append(candidates[pick_global_idx])
                palette_hsv.append(candidates_hsv[pick_global_idx])
                mask[pick_global_idx] = False
            return torch.stack(palette)
        def uniform_palette(img: torch.Tensor, num_colors: int) -> torch.Tensor:
            n = torch.tensor(num_colors ** (1/3), device=img.device)
            steps = int(torch.ceil(n).item())
            lin = torch.linspace(0.0, 1.0, steps, device=img.device)
            r, g, b = torch.meshgrid(lin, lin, lin, indexing="ij")
            grid = torch.stack([r, g, b], dim=-1).reshape(-1, 3)
            return grid[:num_colors]
        def median_cut_palette(img: torch.Tensor, num_colors: int) -> torch.Tensor:
            pixels = img[0].reshape(-1, 3)
            boxes = [pixels]
            while len(boxes) < num_colors:
                ranges = [(b.max(0)[0] - b.min(0)[0]) for b in boxes]
                max_spans = torch.tensor([r.max() for r in ranges], device=img.device)
                i = int(max_spans.argmax().item())
                box = boxes.pop(i)
                chan = int((box.max(0)[0] - box.min(0)[0]).argmax().item())
                sorted_box = box[box[:, chan].argsort()]
                mid = sorted_box.size(0) // 2
                boxes.append(sorted_box[:mid])
                boxes.append(sorted_box[mid:])
            palette = torch.stack([b.mean(0) for b in boxes], dim=0)
            return palette.clamp(0.0, 1.0)
        def consolidate_palette(img: torch.Tensor, num_colors: int, max_candidates: int = 512) -> torch.Tensor:
            pixels = img[0].reshape(-1, 3)
            if pixels.shape[0] > max_candidates:
                idx = torch.randperm(pixels.shape[0], device=pixels.device)[:max_candidates]
                pixels = pixels[idx]
            pixels_i = (pixels * 255.0).round().to(torch.int64)
            uniques = torch.unique(pixels_i, dim=0).to(torch.float32) / 255.0
            palette = uniques.clone()
            while palette.shape[0] > num_colors:
                dists = torch.cdist(palette.unsqueeze(0), palette.unsqueeze(0)).squeeze(0)
                dists.fill_diagonal_(float('inf'))
                min_idx = torch.argmin(dists)
                i = min_idx // dists.shape[1]
                j = min_idx % dists.shape[1]
                merged = (palette[i] + palette[j]) / 2.0
                mask = torch.ones(palette.shape[0], dtype=torch.bool, device=palette.device)
                mask[j] = False
                mask[i] = False
                palette = torch.cat([palette[mask], merged.unsqueeze(0)], dim=0)
            return palette.clamp(0.0, 1.0)
        def generate_palette(img: torch.Tensor,
                            num_colors: int,
                            mode: str) -> torch.Tensor:
            if mode == 'adaptive':
                return kmeans_palette(img, num_colors)
            elif mode == 'median_cut':
                return median_cut_palette(img, num_colors)
            elif mode == 'perceptual':
                return perceptual_palette(img, num_colors)
            elif mode == 'uniform':
                return uniform_palette(img, num_colors)
            elif mode == 'consolidate':
                return consolidate_palette(img, num_colors)
            elif mode == 'none':
                return None
            else:
                raise ValueError(f"Unsupported palette_mode: {mode}")
        def quantize_to_palette(img: torch.Tensor, palette: torch.Tensor, dither: str, threshold_map: torch.Tensor = None, dither_amount: float = 1.0, serpentine: bool = False, threshold_jitter: bool = False) -> torch.Tensor:
            B, C, H, W = img.shape
            device = img.device
            levels = palette.shape[0]
            if dither == "floyd":
                out = img.clone()
                for b in range(B):
                    for y in range(H):
                        l2r = not serpentine or (y % 2 == 0)
                        x_range = range(W) if l2r else range(W - 1, -1, -1)
                        for x in x_range:
                            old = out[b, :, y, x]
                            if threshold_jitter:
                                noise = (torch.rand_like(old) - 0.5) / (levels - 1)
                            else:
                                noise = 0.0
                            old_n = (old + noise).clamp(0.0, 1.0)
                            old_n_ = old_n.unsqueeze(0)
                            dist = torch.sum((palette - old_n_) ** 2, dim=1)
                            idx = torch.argmin(dist)
                            new = palette[idx]
                            out[b, :, y, x] = new
                            err = (old_n - new) * dither_amount
                            if l2r:
                                neighbors = [
                                    ( x + 1, y    , 7 / 16),
                                    ( x - 1, y + 1, 3 / 16),
                                    ( x    , y + 1, 5 / 16),
                                    ( x + 1, y + 1, 1 / 16),
                                ]
                            else:
                                neighbors = [
                                    ( x - 1, y    , 7 / 16),
                                    ( x + 1, y + 1, 3 / 16),
                                    ( x    , y + 1, 5 / 16),
                                    ( x - 1, y + 1, 1 / 16),
                                ]
                            for nx, ny, w in neighbors:
                                if 0 <= nx < W and 0 <= ny < H:
                                    out[b, :, ny, nx] += err * w
                return out.clamp(0.0, 1.0)
            if dither == "noise":
                out = img.clone().clamp(0.0, 1.0)
                B, C, H, W = out.shape
                jitter = (torch.rand_like(out) - 0.5) * dither_amount
                noised = (out + jitter).clamp(0.0, 1.0)
                noised_flat = noised.permute(0, 2, 3, 1).reshape(-1, C)
                dist = torch.cdist(noised_flat.unsqueeze(0), palette.unsqueeze(0)).squeeze(0)
                idx = torch.argmin(dist, dim=1)
                mapped = palette[idx]
                dithered = mapped.view(B, H, W, C).permute(0, 3, 1, 2)
                return dithered.clamp(0.0, 1.0)
            if dither in ("ordered", "pattern"):
                assert threshold_map is not None, "threshold_map required for ordered/pattern"
                bias = threshold_map / levels
                biased = (img + bias).clamp(0.0, 1.0)
                flat = biased.permute(0,2,3,1).reshape(-1, 3)
                dists = torch.cdist(flat, palette.to(device))
                idxs  = dists.argmin(dim=1)
                quant = palette[idxs].reshape(B, H, W, 3) \
                        .permute(0, 3, 1, 2)
                return quant.clamp(0.0, 1.0)
            flat = img.permute(0,2,3,1).reshape(-1, 3)
            dists = torch.cdist(flat, palette.to(device))
            idxs  = dists.argmin(dim=1)
            quant = palette[idxs].reshape(B, H, W, 3) \
                    .permute(0, 3, 1, 2)
            return quant.clamp(0.0, 1.0)
        def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
            r, g, b = img.unbind(dim=1)
            maxc = torch.max(img, dim=1).values
            minc = torch.min(img, dim=1).values
            delta = maxc - minc
            v = maxc
            s = torch.where(maxc > 0, delta / maxc, torch.zeros_like(delta))
            eps_delta = delta + delta.eq(0).float() * 1e-8
            h_ = torch.zeros_like(delta)
            h_ = torch.where(maxc == r, ((g - b) / eps_delta) % 6, h_)
            h_ = torch.where(maxc == g, ((b - r) / eps_delta) + 2, h_)
            h_ = torch.where(maxc == b, ((r - g) / eps_delta) + 4, h_)
            h = (h_ / 6.0) % 1.0
            return torch.stack([h, s, v], dim=1)
        def hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
            h, s, v = img.unbind(dim=1)
            h6 = (h * 6.0)
            i = h6.floor().to(torch.int64) % 6
            f = h6 - i.float()
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            r = torch.where(i == 0, v,
                torch.where(i == 1, q,
                torch.where(i == 2, p,
                torch.where(i == 3, p,
                torch.where(i == 4, t,
                            v)))))
            g = torch.where(i == 0, t,
                torch.where(i == 1, v,
                torch.where(i == 2, v,
                torch.where(i == 3, q,
                torch.where(i == 4, p,
                            p)))))
            b = torch.where(i == 0, p,
                torch.where(i == 1, p,
                torch.where(i == 2, t,
                torch.where(i == 3, v,
                torch.where(i == 4, v,
                            q)))))
            return torch.stack([r, g, b], dim=1)
        assert image.dtype == torch.float32 and image.ndim == 4 and image.shape[-1] == 3
        assert dither in ["none", "floyd", "noise", "ordered", "pattern"]
        assert matrix_size in [2, 4, 8, 16, 32]
        B, H, W, C = image.shape
        img = image.permute(0,3,1,2)
        if isinstance(palette, torch.Tensor) and palette.ndim == 4:
            assert palette.shape[-1] == 3, "Palette image must have 3 channels"
            palette = palette.to(torch.float32)
            palette = palette.reshape(-1, 3)
            palette = palette.unsqueeze(0).expand(B, -1, -1)
        elif palette is None and palette_mode != 'none':
            palettes = []
            for b in range(B):
                single_palette = generate_palette(image[b:b+1], num_colors, palette_mode)
                palettes.append(single_palette.reshape(-1, 3))
            palette = torch.stack(palettes, dim=0)
        else:
            palette = None
        h_small, w_small = int(H / scale), int(W / scale)
        img = F.interpolate(img, size=(h_small, w_small), mode='bilinear', align_corners=False)
        hsv = rgb_to_hsv(img)
        hsv[:, 1] = torch.clamp(hsv[:, 1] * saturation, 0.0, 1.0)
        img = hsv_to_rgb(hsv)
        if dither in ("ordered", "pattern"):
            bayer = make_bayer_matrix(matrix_size, img.device)
            if dither == "pattern":
                bayer = (bayer + (0.5 / (matrix_size * matrix_size))).clamp(0.0, 0.999)
            if matrix_expand_negative:
                if matrix_threshold > 0.0 and palette_mode != 'none':
                    levels = palette.shape[0]
                    matrix_threshold_new = levels * (matrix_threshold ** 1.25)
                    bayer = (bayer - 0.5) * (matrix_threshold_new - (-matrix_threshold_new))
                    bayer = bayer + (matrix_threshold_new + (-matrix_threshold_new)) / 2
                elif matrix_threshold > 0.0 and palette_mode == 'none':
                    matrix_threshold_new = (matrix_threshold ** 1.25)
                    bayer = (bayer - 0.5) * (matrix_threshold_new - (-matrix_threshold_new))
                    bayer = bayer + (matrix_threshold_new + (-matrix_threshold_new)) / 2
            else:
                if matrix_threshold > 0.0 and palette_mode != 'none':
                    levels = palette.shape[0]
                    matrix_threshold_new = levels * (matrix_threshold ** 1.25)
                    bayer = bayer * matrix_threshold_new
                elif matrix_threshold > 0.0 and palette_mode == 'none':
                    matrix_threshold_new = (matrix_threshold ** 1.25)
                    bayer = bayer * matrix_threshold_new
            ty = math.ceil(h_small / matrix_size)
            tx = math.ceil(w_small / matrix_size)
            thresh = bayer.repeat(ty, tx)[:h_small, :w_small]
            threshold_map = thresh.unsqueeze(0).unsqueeze(0)
        out_imgs = []
        for b in range(B):
            img_b = img[b:b+1]
            if palette is not None:
                palette_b = palette[b]
                if dither == "floyd":
                    img_b = quantize_to_palette(img_b, palette_b, dither, None, matrix_threshold / 2, floyd_dither_serpentine, floyd_dither_threshold_jitter)
                elif dither == "noise":
                    img_b = quantize_to_palette(img_b, palette_b, dither, None, matrix_threshold / 2)
                elif dither in ("ordered", "pattern"):
                    img_b = quantize_to_palette(img_b, palette_b, dither, threshold_map=threshold_map)
                else:
                    img_b = quantize_to_palette(img_b, palette_b, dither)
            else:
                if dither == "floyd":
                    img_b = floyd_steinberg_dither(img_b, levels=num_colors, dither_amount=matrix_threshold, serpentine=floyd_dither_serpentine, threshold_jitter=floyd_dither_threshold_jitter)
                elif dither == "noise":
                    img_b = noise_dither(img_b, levels=num_colors, noise_amount=matrix_threshold)
                elif dither in ("ordered", "pattern"):
                    img_b = ordered_dither(img_b, levels=num_colors, threshold_map=threshold_map)
                else:
                    img_b = torch.round(img_b * (num_colors - 1)) / (num_colors - 1)
            out_imgs.append(img_b)
        img = torch.cat(out_imgs, dim=0)
        img = F.interpolate(img, size=(H, W), mode='nearest')
        return (img.permute(0, 2, 3, 1).clamp(0.0, 1.0),)

class Lut:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "lut_filepath": ("STRING", {"default": ""}),
                "chroma_amount": ("FLOAT", {"default": 1.0, "min": 0, "max": 2.0, "step": 0.005}),
                "luma_amount": ("FLOAT", {"default": 1.0, "min": 0, "max": 2.0, "step": 0.005}),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.005}),
                "use_torch_optimized_interpolation": ("BOOLEAN", {"default": False}),
                "interpolation": (
            [   
                'nearest neighbor',
                'bilinear',
                'tricubic',
                'gaussian',
                'stochastic',
            ], {
                "default": 'bilinear'
            }),
                "gaussian_interpolation_radius": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "stochastic_interpolation_samples": ("INT", {"default": 8, "min": 1, "max": 200, "step": 1}),
                "stochastic_interpolation_use_gaussian": ("BOOLEAN", {"default": True}),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 10.0, "step": 0.05}),
                "lut_order": (
            [   
                'RGB',
                'BGR',
            ], {
                "default": 'RGB'
            }),
                "gamma_correction": ("BOOLEAN", {"default": False}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "func_lut"
    CATEGORY = "TrophiHunter/Photography"
    DESCRIPTION = """
A node for applying Color Grading LUTS to an image. This node supports many different types/formats of luts, .1dlut, .png(HALD CLUTs, reshade luts), .3dl, .cube.

lut_filepath = path to the lut file, the path can have "" or not. .1dlut, .png, .3dl and .cube are supported
chroma_amount = changes the chroma amout of the lut (1.0 is default)
luma_amount = changes the luma amout of the lut (1.0 is default)
lut_strength = exponentially scales the lut values
use_torch_optimized_interpolation = use a more torch optimized algorithm while applying the lut. (nearest neighbor and bilinear only)
interpolation = interpolation method to use
gaussian_interpolation_radius = gaussian interpolation blur radius
stochastic_interpolation_samples = how many samples to use for stochastic interpolation
stochastic_interpolation_use_gaussian = use use gaussian blur for stochastic interpolation
sigma = sigma blur for gaussian and stochastic interpolation
lut_order = the order of the lut data, some are rgb and some are bgr
gamma_correction = apply gamma correction to .3dl luts if they are meant for different spaces
opacity = like in photoshop opacity amount of the effect 

"""
    
    def func_lut(self, image, lut_filepath, chroma_amount, luma_amount, lut_strength, use_torch_optimized_interpolation, interpolation, gaussian_interpolation_radius, stochastic_interpolation_samples, stochastic_interpolation_use_gaussian, sigma, lut_order, gamma_correction, opacity):
        def load_1d_lut(filepath: str) -> torch.Tensor:
            data = []
            lut_size = None
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.upper().startswith("LUT_1D_SIZE"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                lut_size = int(parts[1])
                            except ValueError:
                                raise ValueError(f"Invalid LUT size in header: {parts[1]}")
                        continue
                    try:
                        values = list(map(float, line.split()))
                        if values:
                            data.append(values)
                    except ValueError as e:
                        print(f"Warning: Could not parse line '{line}': {e}")
                        continue
            lut_data = np.array(data, dtype=float)
            if lut_size is not None and lut_data.shape[0] != lut_size:
                raise ValueError(f"LUT size mismatch. Header indicates {lut_size} entries but found {lut_data.shape[0]} entries.")
            return torch.from_numpy(lut_data).to(torch.float32)
        def apply_1d_lut_to_image(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
            lut_size = lut.shape[0]
            channels = lut.shape[1]
            ratio = image * (lut_size - 1)
            lower = torch.floor(ratio).to(torch.long)
            frac = ratio - torch.floor(ratio)
            lower = torch.clamp(lower, 0, lut_size - 1)
            upper = torch.clamp(lower + 1, 0, lut_size - 1)
            output = torch.empty_like(image)
            for c in range(channels):
                lut_lower = lut[lower[..., c], c]
                lut_upper = lut[upper[..., c], c]
                output[..., c] = (1 - frac[..., c]) * lut_lower + frac[..., c] * lut_upper
            return output
        def lut_image_to_3d_tensor(path: str) -> tuple[torch.Tensor]:
            img = read_image(path)
            print(img.dtype)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype == torch.uint16:
                img = img.float() / 65535.0
            img = img.permute(1, 2, 0).contiguous()
            H, W, C = img.shape
            if C == 4:
                img = img[:, :, :3]
            if H == W and round((H * W) ** (1/3)) ** 3 == H * W:
                lut_size = round((H * W) ** (1/3))
                lut_3d = img.view(lut_size, lut_size, lut_size, 3)
                return lut_3d
            if W == H * H:
                lut_size = H
                lut_3d = torch.empty((lut_size, lut_size, lut_size, 3), dtype=torch.float32)
                for b in range(lut_size):
                    for g in range(lut_size):
                        for r in range(lut_size):
                            x = r + b * lut_size
                            y = g
                            lut_3d[b, g, r] = img[y, x]
                return lut_3d
            raise ValueError("Input image is not a valid HALD or ReShade LUT")
        def load_cube_lut(filepath: str) -> torch.Tensor:
            lut_size = None
            lut_data = []
            domain_min = 0.0
            domain_max = 1.0
            try:
                with open(filepath, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if not line or line.startswith('#') or line.upper().startswith("TITLE"):
                            continue
                        if line.startswith("LUT_3D_SIZE"):
                            lut_size = int(line.split()[-1])
                        elif line.startswith("DOMAIN_MIN"):
                            domain_min = float(line.split()[-1])
                        elif line.startswith("DOMAIN_MAX"):
                            domain_max = float(line.split()[-1])
                        else:
                            try:
                                values = list(map(float, line.split()))
                                if len(values) == 3:
                                    lut_data.append(values)
                            except ValueError:
                                continue
                if lut_size is None:
                    raise ValueError("Invalid .cube file: Missing LUT_3D_SIZE")
                expected_entries = lut_size ** 3
                if len(lut_data) != expected_entries:
                    raise ValueError(f"Invalid .cube file: Expected {expected_entries} values, but found {len(lut_data)}")
                lut_data = np.array(lut_data, dtype=np.float32).reshape((lut_size, lut_size, lut_size, 3))
                lut_data = np.clip(lut_data, domain_min, domain_max)
                return torch.from_numpy(lut_data)
            except Exception as e:
                raise ValueError(f"Error loading .cube LUT file: {e}")
        def apply_gamma_correction(values: torch.Tensor, input_gamma: str, output_gamma: str) -> torch.Tensor:
            eps = 1e-6
            def to_linear(val: torch.Tensor, gamma: str) -> torch.Tensor:
                val = val.clamp(0.0, 1.0)
                g = gamma.lower()
                if g == "srgb":
                    return torch.where(val <= 0.04045, val / 12.92, ((val + 0.055) / 1.055).pow(2.4))
                elif g in ("gamma22", "gamma2.2"):
                    return val.pow(2.2)
                elif g in ("gamma24", "gamma2.4"):
                    return val.pow(2.4)
                elif g in ("gamma28", "gamma2.8"):
                    return val.pow(2.8)
                elif g == "rec709":
                    return torch.where(val < 0.081, val / 4.5, ((val + 0.099) / 1.099).pow(1/0.45))
                elif g == "logc":
                    return ((10 ** ((val - 0.385537) / 0.2471896) - 0.01) / 5.555556).clamp(0.0, 1.0)
                elif g == "cineon":
                    return ((val - 0.0928) / 0.881).pow(1.0 / 0.6).clamp(0.0, 1.0)
                elif g == "v-log":
                    return torch.where(
                        val < 0.181,
                        (val - 0.125) / 5.6,
                        (torch.pow(10 ** ((val - 0.125) / 0.241514), 1.0) - 0.01) / 5.6
                    ).clamp(0.0, 1.0)
                elif g in ("s-log", "slog"):
                    return (val - 0.092864) / 0.432699
                elif g in ("s-log2", "slog2"):
                    return (10 ** ((val - 0.037584) / 0.432699)) / 100.0
                elif g in ("s-log3", "slog3"):
                    a, b, c = 0.037584, 0.432699, 0.030001222851889303
                    return (torch.pow(10, (val - a) / b) - c).clamp(0.0, 1.0)
                elif g in ("c-log", "clog"):
                    return (10 ** ((val - 0.125) / 0.222)) - 0.01
                elif g in ("c-log2", "clog2"):
                    return (10 ** ((val - 0.1) / 0.225)) - 0.01
                elif g in ("c-log3", "clog3"):
                    return (10 ** ((val - 0.105) / 0.225)) - 0.01
                elif g == "acescg":
                    return val
                elif g == "redlog":
                    return (10 ** ((val - 0.01) / 0.18)) - 0.01
                elif g == "acescct":
                    return torch.where(
                        val < 0.155251141552511,
                        (val - 0.0729055341958355) / 10.5402377416545,
                        torch.pow(2.0, (val - 0.0729055341958355) / 0.2471896) - 0.01
                    ).clamp(0.0, 1.0)
                elif g == "pq":
                    m1, m2 = 2610/16384, 2523/32
                    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
                    val = val.pow(1/m2)
                    return ((val - c1).clamp(min=0) / (c2 - c3 * val)).pow(1/m1).clamp(0.0, 1.0)
                elif g == "hlg":
                    a = 0.17883277
                    b = 1 - 4 * a
                    c = 0.5 - a * torch.log(torch.tensor(4 * a))
                    return torch.where(val <= 0.5, (val ** 2) / 3.0, (torch.exp((val - c) / a) + b) / 12.0)
                elif g == "bmdfilm":
                    return (val - 0.055) / 0.96
                return val
            def from_linear(val: torch.Tensor, gamma: str) -> torch.Tensor:
                val = val.clamp(0.0, 1.0)
                g = gamma.lower()
                if g == "srgb":
                    return torch.where(val <= 0.0031308, val * 12.92, 1.055 * val.pow(1/2.4) - 0.055)
                elif g in ("gamma22", "gamma2.2"):
                    return val.pow(1/2.2)
                elif g in ("gamma24", "gamma2.4"):
                    return val.pow(1/2.4)
                elif g in ("gamma28", "gamma2.8"):
                    return val.pow(1/2.8)
                elif g == "rec709":
                    return torch.where(val < 0.018, val * 4.5, 1.099 * val.pow(0.45) - 0.099)
                elif g == "logc":
                    return (0.2471896 * torch.log10(5.555556 * val + 0.01) + 0.385537).clamp(0.0, 1.0)
                elif g == "cineon":
                    return (0.881 * val.pow(0.6) + 0.0928).clamp(0.0, 1.0)
                elif g == "v-log":
                    return torch.where(
                        val < 0.181,
                        5.6 * val + 0.125,
                        0.241514 * torch.log10(5.6 * val + 0.01) + 0.125
                    ).clamp(0.0, 1.0)
                elif g in ("s-log", "slog"):
                    return (val * 0.432699) + 0.092864
                elif g in ("s-log2", "slog2"):
                    return 0.432699 * torch.log10(val * 100.0) + 0.037584
                elif g in ("s-log3", "slog3"):
                    a, b, c = 0.037584, 0.432699, 0.030001222851889303
                    return (b * torch.log10(val + c)) + a
                elif g in ("c-log", "clog"):
                    return (torch.log10(val + 0.01) * 0.222) + 0.125
                elif g in ("c-log2", "clog2"):
                    return (torch.log10(val + 0.01) * 0.225) + 0.1
                elif g in ("c-log3", "clog3"):
                    return (torch.log10(val + 0.01) * 0.225) + 0.105
                elif g == "acescg":
                    return val
                elif g == "acescct":
                    return torch.where(
                        val < 0.155251141552511,
                        val * 10.5402377416545 + 0.0729055341958355,
                        0.2471896 * torch.log2(val + 0.01) + 0.0729055341958355
                    ).clamp(0.0, 1.0)
                elif g == "redlog":
                    return (10 ** ((val - 0.01) / 0.18)) - 0.01
                elif g == "pq":
                    m1, m2 = 2610/16384, 2523/32
                    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
                    num = c1 + c2 * val.pow(m1)
                    denom = 1 + c3 * val.pow(m1)
                    return (num / denom).pow(m2).clamp(0.0, 1.0)
                elif g == "hlg":
                    a = 0.17883277
                    b = 1 - 4 * a
                    c = 0.5 - a * torch.log(torch.tensor(4 * a))
                    return torch.where(
                        val <= 1/12,
                        torch.sqrt(3 * val),
                        a * torch.log(12 * val - b) + c
                    ).clamp(0.0, 1.0)
                elif g == "bmdfilm":
                    return (val * 0.96 + 0.055).clamp(0.0, 1.0)
                return val
            if input_gamma is None or output_gamma is None or input_gamma.lower() == output_gamma.lower():
                return values.clamp(0.0, 1.0)
            linear = to_linear(values, input_gamma)
            corrected = from_linear(linear, output_gamma)
            return corrected.clamp(0.0, 1.0)
        def load_3dl_lut(lut_path: str, gamma_correction=False) -> torch.Tensor:
            input_gamma = None
            output_gamma = None
            size = None
            lut_values = []
            with open(lut_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    if "INPUTGAMMA" in line:
                        input_gamma = line.split("INPUTGAMMA")[-1].strip()
                    elif "OUTPUTGAMMA" in line:
                        output_gamma = line.split("OUTPUTGAMMA")[-1].strip()
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[0].upper() == "LUT_3D_SIZE":
                    size = int(parts[1])
                    continue
                if size is None and all(p.isdigit() for p in parts):
                    size = len(parts)
                    continue
                if len(parts) == 3:
                    try:
                        rgb = [float(v) for v in parts]
                        lut_values.append(rgb)
                    except ValueError:
                        raise ValueError(f"Invalid RGB values in LUT file: {line}")
            if not lut_values:
                raise ValueError("No LUT data found in .3dl file.")
            if size is None:
                size = round(len(lut_values) ** (1/3))
            if len(lut_values) != size ** 3:
                raise ValueError(f"Unexpected LUT size: expected {size**3} values, got {len(lut_values)}.")
            lut_array = np.array(lut_values, dtype=np.float32).reshape((size, size, size, 3))
            max_val = np.max(lut_array)
            if max_val > 1.0:
                lut_array /= max_val
            lut_tensor = torch.from_numpy(lut_array)
            if gamma_correction:
                if input_gamma and output_gamma and input_gamma != output_gamma:
                    lut_tensor = apply_gamma_correction(lut_tensor, input_gamma=input_gamma, output_gamma=output_gamma)
            return lut_tensor
        def round_005(x: float) -> float:
            return round(x / 0.005) * 0.005
        def apply_3d_lut_to_image(image: torch.Tensor, lut: torch.Tensor,
                                lut_order: str,
                                lut_strength: float = 1.0,
                                opacity: int = 100,
                                chroma_amount: float = 1.0, 
                                luma_amount: float = 1.0) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            if lut_order.upper() == "RGB":
                image_order = image[..., [0, 1, 2]]
            elif lut_order.upper() == "BGR":
                image_order = image[..., [2, 1, 0]]
            opacity_scale = opacity * 0.01
            lut_size = lut.shape[0]
            img_scaled = image_order * (lut_size - 1)
            if lut_strength != 1.0:
                img_scaled = torch.pow(img_scaled, lut_strength)
            img_scaled = torch.clamp(img_scaled, 0, lut_size - 1)
            i0 = torch.floor(img_scaled).to(torch.long)
            i1 = torch.clamp(i0 + 1, max=lut_size - 1)
            d = img_scaled - torch.floor(img_scaled)
            mapped_image = torch.zeros_like(image)
            lut_indices = (0, 1, 2)
            def lerp(a, b, w):
                return a * (1 - w) + b * w
            for ch in range(3):
                c000 = lut[i0[..., 0], i0[..., 1], i0[..., 2], lut_indices[ch]]
                c001 = lut[i1[..., 0], i0[..., 1], i0[..., 2], lut_indices[ch]]
                c010 = lut[i0[..., 0], i1[..., 1], i0[..., 2], lut_indices[ch]]
                c011 = lut[i1[..., 0], i1[..., 1], i0[..., 2], lut_indices[ch]]
                c100 = lut[i0[..., 0], i0[..., 1], i1[..., 2], lut_indices[ch]]
                c101 = lut[i1[..., 0], i0[..., 1], i1[..., 2], lut_indices[ch]]
                c110 = lut[i0[..., 0], i1[..., 1], i1[..., 2], lut_indices[ch]]
                c111 = lut[i1[..., 0], i1[..., 1], i1[..., 2], lut_indices[ch]]
                c00 = lerp(c000, c001, d[..., 0])
                c01 = lerp(c010, c011, d[..., 0])
                c10 = lerp(c100, c101, d[..., 0])
                c11 = lerp(c110, c111, d[..., 0])
                c0 = lerp(c00, c01, d[..., 1])
                c1 = lerp(c10, c11, d[..., 1])
                mapped_image[..., ch] = lerp(c0, c1, d[..., 2])
            mapped_image = torch.clamp(mapped_image, 0.0, 1.0)
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (mapped_image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = mapped_image - lut_luma
                mapped_image = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                mapped_image = mapped_image
            if opacity_scale != 1.0:
                blended = (1 - opacity_scale) * image + opacity_scale * mapped_image
                return torch.clamp(blended, 0.0, 1.0)
            else:
                return mapped_image
        def apply_3d_lut_grid_sample(image: torch.Tensor, lut: torch.Tensor, interpolation, lut_order, lut_strength, opacity, chroma_amount, luma_amount) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            image_normalized = image
            opacity_scale = opacity * 0.01
            H, W, C = image_normalized.shape
            assert C == 3, "Image must have 3 channels (RGB)"
            lut_size = lut.shape[0]
            if lut_order.upper() == "RGB":
                r = image_normalized[:, :, 0]
                g = image_normalized[:, :, 1]
                b = image_normalized[:, :, 2]
            elif lut_order.upper() == "BGR":
                r = image_normalized[:, :, 2]
                g = image_normalized[:, :, 1]
                b = image_normalized[:, :, 0]
            r_scaled = r * (lut_size - 1)
            g_scaled = g * (lut_size - 1)
            b_scaled = b * (lut_size - 1)
            if lut_strength != 1.0:
                r_scaled = torch.pow(r_scaled, lut_strength)
                g_scaled = torch.pow(g_scaled, lut_strength)
                b_scaled = torch.pow(b_scaled, lut_strength)
            r_scaled = torch.clamp(r_scaled, 0, lut_size - 1)
            g_scaled = torch.clamp(g_scaled, 0, lut_size - 1)
            b_scaled = torch.clamp(b_scaled, 0, lut_size - 1)
            lut_vol = lut.permute(3, 0, 1, 2).unsqueeze(0)
            grid = torch.stack([b_scaled, g_scaled, r_scaled], dim=-1)
            grid = grid / (lut_size - 1) * 2 - 1
            grid = grid.unsqueeze(0).unsqueeze(0)
            if interpolation == 'nearest neighbor':
                output = F.grid_sample(lut_vol, grid, mode='nearest', align_corners=True)
            else:
                output = F.grid_sample(lut_vol, grid, mode='bilinear', align_corners=True)
            if output.dim() == 5:
                output = output[0, :, 0]
            else:
                output = output[0]
            output = output.permute(1, 2, 0)
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (output @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = output - lut_luma
                output = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                output = output
            if opacity_scale != 1.0:
                blended = (1 - opacity_scale) * image + opacity_scale * output
                return torch.clamp(blended, 0.0, 1.0)
            else:
                return output
        def apply_3d_lut_nearest_neighbor(image: torch.Tensor, lut: torch.Tensor, lut_order, lut_strength, opacity, chroma_amount, luma_amount) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            H, W, C = image.shape
            assert C == 3, "Image must have 3 channels (RGB)"
            lut_size = lut.shape[0]
            assert lut.shape == (lut_size, lut_size, lut_size, 3), "LUT must be [L, L, L, 3]"
            if lut_order.upper() == "RGB":
                r = image[:, :, 0]
                g = image[:, :, 1]
                b = image[:, :, 2]
            elif lut_order.upper() == "BGR":
                r = image[:, :, 2]
                g = image[:, :, 1]
                b = image[:, :, 0]
            else:
                raise ValueError("lut_order must be 'RGB' or 'BGR'")
            r_scaled = r * (lut_size - 1)
            g_scaled = g * (lut_size - 1)
            b_scaled = b * (lut_size - 1)
            if lut_strength != 1.0:
                r_scaled = torch.pow(r_scaled, lut_strength)
                g_scaled = torch.pow(g_scaled, lut_strength)
                b_scaled = torch.pow(b_scaled, lut_strength)
            idx_r = torch.round(r_scaled).long().clamp(0, lut_size - 1)
            idx_g = torch.round(g_scaled).long().clamp(0, lut_size - 1)
            idx_b = torch.round(b_scaled).long().clamp(0, lut_size - 1)
            output = lut[idx_r, idx_g, idx_b]
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (output @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = output - lut_luma
                output = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                output = output
            opacity_scale = opacity * 0.01
            if opacity_scale != 1.0:
                output = (1.0 - opacity_scale) * image + opacity_scale * output
            return torch.clamp(output, 0.0, 1.0)
        def cubic_kernel(x: torch.Tensor) -> torch.Tensor:
            abs_x = torch.abs(x)
            abs_x2 = abs_x ** 2
            abs_x3 = abs_x ** 3
            a = -0.5
            k = ( (a + 2) * abs_x3 - (a + 3) * abs_x2 + 1 ) * (abs_x <= 1).float() + \
                ( a * abs_x3 - 5*a * abs_x2 + 8*a * abs_x - 4*a ) * ((abs_x > 1) & (abs_x < 2)).float()
            return k
        def apply_3d_lut_tricubic(image: torch.Tensor, lut: torch.Tensor, lut_order, lut_strength, opacity, chroma_amount, luma_amount) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            H, W, C = image.shape
            assert C == 3, "Image must have 3 channels (RGB)"
            L = lut.shape[0]
            assert lut.shape == (L, L, L, 3), "LUT must be [L, L, L, 3]"
            if lut_order.upper() == "RGB":
                r = image[:, :, 0] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 2] * (L - 1)
            elif lut_order.upper() == "BGR":
                r = image[:, :, 2] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 0] * (L - 1)
            if lut_strength != 1.0:
                r = torch.pow(r, lut_strength)
                g = torch.pow(g, lut_strength)
                b = torch.pow(b, lut_strength)
            r_base = torch.floor(r).long()
            g_base = torch.floor(g).long()
            b_base = torch.floor(b).long()
            r_frac = r - r_base.float()
            g_frac = g - g_base.float()
            b_frac = b - b_base.float()
            output = torch.zeros((H, W, 3), device=image.device)
            for i in range(-1, 3):
                for j in range(-1, 3):
                    for k in range(-1, 3):
                        idx_r = (r_base + i).clamp(0, L - 1)
                        idx_g = (g_base + j).clamp(0, L - 1)
                        idx_b = (b_base + k).clamp(0, L - 1)
                        w_r = cubic_kernel(i - r_frac)
                        w_g = cubic_kernel(j - g_frac)
                        w_b = cubic_kernel(k - b_frac)
                        weight = (w_r * w_g * w_b).unsqueeze(-1)
                        lut_val = lut[idx_r, idx_g, idx_b]
                        output += weight * lut_val
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (output @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = output - lut_luma
                output = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                output = output
            opacity_scale = opacity * 0.01
            if opacity_scale != 1.0:
                output = (1.0 - opacity_scale) * image + opacity_scale * output
            return torch.clamp(output, 0.0, 1.0)
        def apply_3d_lut_gaussian(image: torch.Tensor, lut: torch.Tensor, lut_order, lut_strength, opacity, chroma_amount, luma_amount, radius: int = 2, sigma: float = 0.5) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            H, W, C = image.shape
            L = lut.shape[0]
            assert C == 3 and lut.shape == (L, L, L, 3), "Invalid image or LUT shape"
            if lut_order.upper() == "RGB":
                r = image[:, :, 0] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 2] * (L - 1)
            elif lut_order.upper() == "BGR":
                r = image[:, :, 2] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 0] * (L - 1)
            if lut_strength != 1.0:
                r = torch.pow(r, lut_strength)
                g = torch.pow(g, lut_strength)
                b = torch.pow(b, lut_strength)
            r0 = torch.floor(r).long()
            g0 = torch.floor(g).long()
            b0 = torch.floor(b).long()
            output = torch.zeros((H, W, 3), dtype=torch.float32, device=image.device)
            weight_sum = torch.zeros((H, W, 1), dtype=torch.float32, device=image.device)
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    for k in range(-radius, radius + 1):
                        idx_r = (r0 + i).clamp(0, L - 1)
                        idx_g = (g0 + j).clamp(0, L - 1)
                        idx_b = (b0 + k).clamp(0, L - 1)
                        dr = r - idx_r.float()
                        dg = g - idx_g.float()
                        db = b - idx_b.float()
                        dist_sq = dr**2 + dg**2 + db**2
                        weight = torch.exp(-dist_sq / (2 * sigma ** 2)).unsqueeze(-1)
                        lut_val = lut[idx_r, idx_g, idx_b]
                        output += weight * lut_val
                        weight_sum += weight
            output /= weight_sum.clamp(min=1e-8)
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (output @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = output - lut_luma
                output = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                output = output
            opacity_scale = opacity * 0.01
            if opacity_scale != 1.0:
                output = (1.0 - opacity_scale) * image + opacity_scale * output
            return output.clamp(0, 1)
        def apply_3d_lut_monte_carlo(
            image: torch.Tensor,
            lut: torch.Tensor,
            lut_order,
            lut_strength,
            opacity,
            chroma_amount,
            luma_amount,
            num_samples: int = 8,
            sigma: float = 0.5,
            use_gaussian: bool = True
        ) -> torch.Tensor:
            lut_strength = round_005(lut_strength)
            chroma_amount = round_005(chroma_amount)
            luma_amount = round_005(luma_amount)
            H, W, C = image.shape
            L = lut.shape[0]
            assert C == 3 and lut.shape == (L, L, L, 3)
            if lut_order.upper() == "RGB":
                r = image[:, :, 0] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 2] * (L - 1)
            elif lut_order.upper() == "BGR":
                r = image[:, :, 2] * (L - 1)
                g = image[:, :, 1] * (L - 1)
                b = image[:, :, 0] * (L - 1)
            if lut_strength != 1.0:
                r = torch.pow(r, lut_strength)
                g = torch.pow(g, lut_strength)
                b = torch.pow(b, lut_strength)
            output = torch.zeros((H, W, 3), dtype=torch.float32, device=image.device)
            for _ in range(num_samples):
                if use_gaussian:
                    dr = torch.randn((H, W), device=image.device) * sigma
                    dg = torch.randn((H, W), device=image.device) * sigma
                    db = torch.randn((H, W), device=image.device) * sigma
                else:
                    dr = (torch.rand((H, W), device=image.device) * 2 - 1) * sigma
                    dg = (torch.rand((H, W), device=image.device) * 2 - 1) * sigma
                    db = (torch.rand((H, W), device=image.device) * 2 - 1) * sigma
                idx_r = (r + dr).round().clamp(0, L - 1).long()
                idx_g = (g + dg).round().clamp(0, L - 1).long()
                idx_b = (b + db).round().clamp(0, L - 1).long()
                lut_val = lut[idx_r, idx_g, idx_b]
                output += lut_val
            output /= num_samples
            if chroma_amount != 1.0 or luma_amount != 1.0:
                orig_luma = (image @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                lut_luma = (output @ torch.tensor([0.299, 0.587, 0.114], device=image.device)).unsqueeze(-1)
                orig_chroma = image - orig_luma
                lut_chroma = output - lut_luma
                output = (
                    luma_amount * orig_luma + (1 - luma_amount) * lut_luma +
                    chroma_amount * lut_chroma + (1 - chroma_amount) * orig_chroma
                )
            else:
                output = output
            opacity_scale = opacity * 0.01
            if opacity_scale != 1.0:
                output = (1.0 - opacity_scale) * image + opacity_scale * output
            return output.clamp(0, 1)
        B = image.shape[0]
        out_tensor = torch.empty_like(image)
        lut_filepath = lut_filepath.strip('\'"')
        ext = os.path.splitext(lut_filepath)[1].lower()
        if ext == '.1dlut':
            lut_1d = load_1d_lut(lut_filepath)
            for idx in range(B):
                out_tensor[idx] = apply_1d_lut_to_image(image[idx], lut_1d)
        elif ext == '.png':
            lut_3d = lut_image_to_3d_tensor(lut_filepath)
            lut_3d = lut_3d.to(torch.float32)
        elif ext == '.cube':
            lut_3d = load_cube_lut(lut_filepath)
            lut_3d = lut_3d.to(torch.float32)
        elif ext == '.3dl':
            lut_3d = load_3dl_lut(lut_filepath, gamma_correction)
            lut_3d = lut_3d.to(torch.float32)
        else:
            raise ValueError(f"Unsupported LUT extension: {ext}")
        if interpolation == 'nearest neighbor':
            if use_torch_optimized_interpolation:
                for idx in range(B):
                    out_tensor[idx] = apply_3d_lut_grid_sample(image[idx], lut_3d, interpolation, lut_order=lut_order, lut_strength=lut_strength, opacity=opacity, chroma_amount=chroma_amount, luma_amount=luma_amount)
            else:
                for idx in range(B):
                    out_tensor[idx] = apply_3d_lut_nearest_neighbor(image[idx], lut_3d, lut_order=lut_order, lut_strength=lut_strength, opacity=opacity, chroma_amount=chroma_amount, luma_amount=luma_amount)
        elif interpolation == 'bilinear':
            if use_torch_optimized_interpolation:
                for idx in range(B):
                    out_tensor[idx] = apply_3d_lut_grid_sample(image[idx], lut_3d, interpolation, lut_order=lut_order, lut_strength=lut_strength, opacity=opacity, chroma_amount=chroma_amount, luma_amount=luma_amount)
            else:
                for idx in range(B):
                    out_tensor[idx] = apply_3d_lut_to_image(image[idx], lut_3d,
                                                            lut_order=lut_order,
                                                            lut_strength=lut_strength,
                                                            opacity=opacity,
                                                            chroma_amount=chroma_amount, 
                                                            luma_amount=luma_amount)
        elif interpolation == 'tricubic':
            for idx in range(B):
                out_tensor[idx] = apply_3d_lut_tricubic(image[idx], lut_3d,
                                                        lut_order=lut_order,
                                                        lut_strength=lut_strength,
                                                        opacity=opacity,
                                                        chroma_amount=chroma_amount,
                                                        luma_amount=luma_amount)
        elif interpolation == 'gaussian':
            for idx in range(B):
                out_tensor[idx] = apply_3d_lut_gaussian(image[idx], lut_3d,
                                                        lut_order=lut_order,
                                                        lut_strength=lut_strength,
                                                        opacity=opacity,
                                                        chroma_amount=chroma_amount,
                                                        luma_amount=luma_amount,
                                                        radius=gaussian_interpolation_radius,
                                                        sigma=sigma)
        elif interpolation == 'stochastic':
            for idx in range(B):
                out_tensor[idx] = apply_3d_lut_monte_carlo(image[idx], lut_3d,
                                                        lut_order=lut_order,
                                                        lut_strength=lut_strength,
                                                        opacity=opacity,
                                                        chroma_amount=chroma_amount,
                                                        luma_amount=luma_amount,
                                                        num_samples=stochastic_interpolation_samples,
                                                        sigma=sigma,
                                                        use_gaussian=stochastic_interpolation_use_gaussian)
        else:
            for idx in range(B):
                out_tensor[idx] = apply_3d_lut_to_image(image[idx], lut_3d,
                                                        lut_order=lut_order,
                                                        lut_strength=lut_strength,
                                                        opacity=opacity,
                                                        chroma_amount=chroma_amount,
                                                        luma_amount=luma_amount)
        return (out_tensor,)