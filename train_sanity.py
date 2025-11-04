import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# --------------------
# 1. Simple UNet Generator
# --------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=4):
        super().__init__()
        assert depth >= 1, "depth must be >=1"
        self.depth = depth
        self.encs = nn.ModuleList()
        self.encoder_channels = []

        prev_ch = in_channels
        # encoder: double channels each step
        for i in range(depth):
            out_ch = base_features * (2**i)
            if i == 0:
                block = nn.Sequential(nn.Conv2d(prev_ch, out_ch, 4, 2, 1), nn.ReLU(0.2))
            else:
                block = nn.Sequential(nn.Conv2d(prev_ch, out_ch, 4, 2, 1),
                                      nn.InstanceNorm2d(out_ch), nn.ReLU(0.2))
            self.encs.append(block)
            self.encoder_channels.append(out_ch)
            prev_ch = out_ch

        # decoder: we construct layers so that each ConvTranspose2d maps:
        # current_in -> next_out where next_out equals the encoder channel of the layer we will concat with.
        self.decs = nn.ModuleList()
        cur_in = self.encoder_channels[-1]  # bottom channels
        # iterate backwards over encoder channels (skip the bottom one), for each create ConvTranspose2d(cur_in, out_ch)
        for enc_ch in reversed(self.encoder_channels[:-1]):
            out_ch = enc_ch
            block = nn.Sequential(
                nn.ConvTranspose2d(cur_in, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )
            self.decs.append(block)
            # after concat, next cur_in becomes out_ch * 2
            cur_in = out_ch * 2

        # final layer: after last concat, cur_in is base_features*2 (since we concat with first encoder output)
        self.final = nn.ConvTranspose2d(cur_in, out_channels, 4, 2, 1)
        self.out_act = nn.Sigmoid()
        # self.out_act = nn.Tanh()

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        # decode
        for i, dec in enumerate(self.decs):
            out = dec(out)
            skip = skips[-(i+2)]  # match encoder
            if out.shape[2:] != skip.shape[2:]:
                out = nn.functional.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
        out = self.final(out)
        return self.out_act(out)


# --------------------
# 2. Load one input/target pair
# --------------------
to_tensor = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_image(path, gray=False):
    img = Image.open(path).convert('L' if gray else 'RGB')
    return to_tensor(img).unsqueeze(0)

# Replace these with your file paths:
input_path = "input.png"   # sketch
target_path = "target.png" # colored ground truth
x = load_image(input_path, gray=True)
y = load_image(target_path, gray=False)

# --------------------
# 3. Train loop
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
G = UNetGenerator().to(device)
x, y = x.to(device), y.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(G.parameters(), lr=1e-3)

losses = []
for step in range(200):
    optimizer.zero_grad()
    pred = G(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"step {step:04d} | loss {loss.item():.6f}")
    if step % 50 == 0:
        with torch.no_grad():
            out = G(x).clamp(0, 1)
            plt.imshow(out[0].permute(1,2,0).cpu())
            plt.title(f"Step {step}")
            plt.show()

# --------------------
# 4. Plot loss curve and save output
# --------------------
plt.plot(losses)
plt.title("Overfit test loss")
plt.show()

with torch.no_grad():
    out = G(x).clamp(0, 1)
    plt.subplot(1,3,1); plt.imshow(x[0,0].cpu(), cmap='gray'); plt.title("Input")
    plt.subplot(1,3,2); plt.imshow(y[0].permute(1,2,0).cpu()); plt.title("Target")
    plt.subplot(1,3,3); plt.imshow(out[0].permute(1,2,0).cpu()); plt.title("Output")
    plt.show()
