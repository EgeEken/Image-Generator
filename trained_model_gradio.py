import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr
from PIL import Image
import numpy as np

# -----------------------------
# 1. UNet generator (GroupNorm stable)
# -----------------------------
def _group_norm(channels):
    # choose groups: min(32, channels) but must divide channels. Choose divisor near 32.
    groups = min(32, channels)
    # reduce groups until divides channels
    while channels % groups != 0:
        groups -= 1
        if groups == 1:
            break
    return nn.GroupNorm(groups, channels)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=4, use_norm=True):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.depth = depth
        self.use_norm = use_norm
        self.encs = nn.ModuleList()
        self.encoder_channels = []

        prev_ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2**i)
            layers = [nn.Conv2d(prev_ch, out_ch, 4, 2, 1)]
            # first encoder typically no norm, use LeakyReLU
            if use_norm and i != 0:
                layers.append(_group_norm(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.encs.append(nn.Sequential(*layers))
            self.encoder_channels.append(out_ch)
            prev_ch = out_ch

        # decoder
        self.decs = nn.ModuleList()
        cur_in = self.encoder_channels[-1]
        for enc_ch in reversed(self.encoder_channels[:-1]):
            out_ch = enc_ch
            layers = [nn.ConvTranspose2d(cur_in, out_ch, 4, 2, 1)]
            if use_norm:
                layers.append(_group_norm(out_ch))
            layers.append(nn.ReLU(inplace=True))
            self.decs.append(nn.Sequential(*layers))
            cur_in = out_ch * 2

        self.final = nn.ConvTranspose2d(cur_in, out_channels, 4, 2, 1)
        self.out_act = nn.Sigmoid()  # because we use 0..1 targets

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        for i, dec in enumerate(self.decs):
            out = dec(out)
            skip = skips[-(i+2)]
            if out.shape[2:] != skip.shape[2:]:
                out = nn.functional.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
        out = self.final(out)
        return self.out_act(out)

# --------------------------------------------------------
# 2. Load the trained generator
# --------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "valid_generator_model.pth"  # adjust if needed
G = UNetGenerator().to(device)
state_dict = torch.load(model_path, map_location=device)
G.load_state_dict(state_dict, strict=True)
G.eval()

# --------------------------------------------------------
# 3. Preprocessing utilities
# --------------------------------------------------------
to_tensor = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

# --------------------------------------------------------
# 4. Inference function for Gradio
# --------------------------------------------------------
def generate_from_sketch(img):
    """
    img: PIL image from Gradio's sketchpad (RGBA or RGB)
    """
    # Resize and convert to grayscale 0–1 tensor
    rgba = Image.fromarray(img["layers"][0]).convert("RGBA")
    black_bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    rgba_on_black = Image.alpha_composite(black_bg, rgba)

    # show the image for debugging
    #rgba_on_black.show()

    # Convert to grayscale and resize for the model
    gray = rgba_on_black.convert("L").resize((128, 128))
    x = to_tensor(gray).unsqueeze(0).to(device)

    with torch.no_grad():
        out = G(x).squeeze(0).cpu().clamp(0, 1)

    # Convert tensor → PIL
    out_img = to_pil(out)
    return out_img

# --------------------------------------------------------
# 5. Gradio Interface
# --------------------------------------------------------
interface = gr.Interface(
    fn=generate_from_sketch,
    inputs=gr.Sketchpad(
        label="Draw your sketch",
        brush=gr.Brush(default_size=2, default_color="#FFFFFF", colors=["#FFFFFF"]),
        image_mode="RGBA",
    ),
    outputs=gr.Image(label="Generated Image", type="pil"),
    title="Sketch-to-Image Generator",
    description="Draw a binary (black & white) sketch below and let the trained generator colorize it.",
    allow_flagging="never",
    live=True,
)

if __name__ == "__main__":
    interface.launch(share=True)
