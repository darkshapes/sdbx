import math

import torch

from sdbx import config
from sdbx.nodes.types import *

@node
def latent_from_batch(
    samples: Latent,
    batch_index: A[int, Numerical(min=0, max=63, step=1)] = 0,
    length: A[int, Numerical(min=1, max=64, step=1)] = 1
) -> Latent:
        s = samples.copy()
        s_in = samples["samples"]
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s["samples"] = s_in[batch_index:batch_index + length].clone()
        if "noise_mask" in samples:
            masks = samples["noise_mask"]
            if masks.shape[0] == 1:
                s["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < s_in.shape[0]:
                    masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
                s["noise_mask"] = masks[batch_index:batch_index + length].clone()
        if "batch_index" not in s:
            s["batch_index"] = [x for x in range(batch_index, batch_index+length)]
        else:
            s["batch_index"] = samples["batch_index"][batch_index:batch_index + length]
        return (s,)

@node
def repeat_latent_batch(
    samples: Latent,
    amount: A[int, Numerical(min=1, max=64, step=1)] = 1,
) -> Latent:
        s = samples.copy()
        s_in = samples["samples"]

        s["samples"] = s_in.repeat((amount, 1,1,1))
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[:s_in.shape[0]]
            s["noise_mask"] = samples["noise_mask"].repeat((amount, 1,1,1))
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            s["batch_index"] = s["batch_index"] + [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
        return (s,)

@node
def latent_upscale(
    samples: Latent,
    upscale_method: Literal["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
    width: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 512,
    height: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 512,
    crop: Literal["disabled", "center"] = "disabled"
) -> Latent:
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()

            if width == 0:
                height = max(64, height)
                width = max(64, round(samples["samples"].shape[3] * height / samples["samples"].shape[2]))
            elif height == 0:
                width = max(64, width)
                height = max(64, round(samples["samples"].shape[2] * width / samples["samples"].shape[3]))
            else:
                width = max(64, width)
                height = max(64, height)

            s["samples"] = utils.common_upscale(samples["samples"], width // 8, height // 8, upscale_method, crop)
        return (s,)

@node
def latent_upscale_by(
    samples: Latent,
    upscale_method: Literal["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
    scale_by: A[float, Numerical(min=0.01, max=8.0, step=0.01)] = 1.5
) -> Latent:
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
        return (s,)

@node
def latent_rotate(
    samples: Latent,
    rotation: Literal["none", "90 degrees", "180 degrees", "270 degrees"],
) -> Latent:
        s = samples.copy()
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        s["samples"] = torch.rot90(samples["samples"], k=rotate_by, dims=[3, 2])
        return (s,)


@node
def latent_flip(
    samples: Latent,
    flip_method: Literal["x-axis: vertically", "y-axis: horizontally"],
) -> Latent:
        s = samples.copy()
        if flip_method.startswith("x"):
            s["samples"] = torch.flip(samples["samples"], dims=[2])
        elif flip_method.startswith("y"):
            s["samples"] = torch.flip(samples["samples"], dims=[3])

        return (s,)

@node
def latent_composite(
    samples_to: Latent,
    samples_from: Latent,
    x: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    y: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    feather: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
) -> Latent:
    # exdysa - origin below code was this
    #  def composite(self, samples_to, samples_from, x, y, composite_method="normal", feather=0):
    x =  x // 8
    y = y // 8
    feather = feather // 8
    samples_out = samples_to.copy()
    s = samples_to["samples"].clone()
    samples_to = samples_to["samples"]
    samples_from = samples_from["samples"]
    if feather == 0:
        s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
    else:
        samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        mask = torch.ones_like(samples_from)
        for t in range(feather):
            if y != 0:
                mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

            if y + samples_from.shape[2] < samples_to.shape[2]:
                mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
            if x != 0:
                mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
            if x + samples_from.shape[3] < samples_to.shape[3]:
                mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
        rev_mask = torch.ones_like(mask) - mask
        s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask
    samples_out["samples"] = s
    return (samples_out,)

@node
def latent_blend(
    samples1: Latent,
    samples2: Latent,
    blend_factor: A[float, Numerical(min=0, max=1.0, step=0.01)] = 0.5,
) -> Latent:
    # exdysa - origin code was so
    # def blend(self, samples1, samples2, blend_factor:float, blend_mode: str="normal"):
    samples_out = samples1.copy()
    samples1 = samples1["samples"]
    samples2 = samples2["samples"]

    if samples1.shape != samples2.shape:
        samples2.permute(0, 3, 1, 2)
        samples2 = utils.common_upscale(samples2, samples1.shape[3], samples1.shape[2], 'bicubic', crop='center')
        samples2.permute(0, 2, 3, 1)

    samples_blended = self.blend_mode(samples1, samples2, blend_mode)
    samples_blended = samples1 * blend_factor + samples_blended * (1 - blend_factor)
    samples_out["samples"] = samples_blended
    return (samples_out,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

@node
def latent_crop(
    samples: Latent,
    width: A[int, Numerical(min=64, max=MAX_RESOLUTION, step=8)] = 64,
    height: A[int, Numerical(min=64, max=MAX_RESOLUTION, step=8)] = 64,
    x: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    y: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
) -> Latent:
        s = samples.copy()
        samples = samples['samples']
        x =  x // 8
        y = y // 8

        #enfonce minimum size of 64
        if x > (samples.shape[3] - 8):
            x = samples.shape[3] - 8
        if y > (samples.shape[2] - 8):
            y = samples.shape[2] - 8

        new_height = height // 8
        new_width = width // 8
        to_x = new_width + x
        to_y = new_height + y
        s['samples'] = samples[:,:,y:to_y, x:to_x]
        return (s,)

@node
def set_latent_noise_mask(
    samples: Latent,
    mask: Mask,
) -> Latent:
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return (s,)

@node
def image_scale(
    pixels: Image,
    upscale_method: Literal["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
    crop: Literal["disabled", "center"],
    width: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=1)] = 512,
    height: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=1)] = 512,
) -> Image:
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)

@node
def image_scale_by(
    pixels: Image,
    upscale_method: Literal["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
    scale_by: A[float, Numerical(min=0.01, max=8.0, step=0.01)]= 1.0
) -> Image:

        samples = image.movedim(-1,1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

@node
def image_invert(
    image: Image
) -> Image:
        s = 1.0 - image
        return (s,)

@node
def image_batch(
    image1: Image,
    image2: Image,
 ) -> Image:
        if image1.shape[1:] != image2.shape[1:]:
            image2 = utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        s = torch.cat((image1, image2), dim=0)
        return (s,)

@node
def image_pad_for_outpaint(
    pixels: Image,
    left: A[int, Numerical(min=-0, max=MAX_RESOLUTION, step=8)] = 0,
    top: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    right: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    bottom: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=8)] = 0,
    feathering: A[int, Numerical(min=0, max=MAX_RESOLUTION, step=1)] = 40,
) -> Tuple[Image, Mask]:
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)