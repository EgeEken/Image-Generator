#!/usr/bin/env python3
"""
tk_sketch_generator.py
Simple Tkinter sketchpad + preview for a sketch->image generator.

Features:
- Left: sketch canvas (draw black on white)
- Right: preview updated in real-time (debounced) using a loaded PyTorch model
- Buttons: Brush, Eraser, Undo, Redo, Clear, Load Model, Generate, Real-time toggle
- Undo/Redo implemented by storing PIL image history
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageOps
import numpy as np
import threading
import time
import os

# Optional: torch for model inference
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

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

# -----------------------
# App configuration
# -----------------------
CANVAS_PIX = 512       # on-screen canvas size
MODEL_SIZE = 128       # input size to model (128x128)
BRUSH_DEFAULT = 12
HISTORY_LIMIT = 30     # undo/redo stack limit
PREVIEW_DEBOUNCE_MS = 250  # milliseconds debounce for real-time updates

# -----------------------
# Tkinter App
# -----------------------
class SketchApp:
    def __init__(self, root):
        self.root = root
        root.title("Sketch -> Generator (local Tkinter)")

        # Left: drawing canvas (PIL image under the hood)
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.LEFT, padx=6, pady=6)

        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_PIX, height=CANVAS_PIX, bg='white', bd=2, relief=tk.SUNKEN)
        self.canvas.pack()

        # PIL image we actually draw on (L mode)
        self.image = Image.new("L", (CANVAS_PIX, CANVAS_PIX), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # PhotoImage for canvas display
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Right: preview
        self.preview_frame = tk.Frame(root)
        self.preview_frame.pack(side=tk.RIGHT, padx=6, pady=6, fill=tk.BOTH, expand=False)

        self.preview_label = tk.Label(self.preview_frame, text="Preview", font=("Arial", 12))
        self.preview_label.pack()

        self.preview_canvas = tk.Canvas(self.preview_frame, width=CANVAS_PIX//2, height=CANVAS_PIX//2, bg='lightgray', bd=2, relief=tk.SUNKEN)
        self.preview_canvas.pack()

        self.preview_image = Image.new("RGB", (CANVAS_PIX//2, CANVAS_PIX//2), color=(200,200,200))
        self.preview_photo = ImageTk.PhotoImage(self.preview_image)
        self.preview_canvas_img = self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)

        # Controls (center bottom right below the canvas)
        controls_frame = tk.Frame(root)
        controls_frame.pack(side=tk.BOTTOM, pady=6, fill=tk.X, expand=False)

        # Tool buttons placed centered bottom-right under the drawing area
        self.tool_frame = tk.Frame(controls_frame)
        self.tool_frame.pack(side=tk.RIGHT, padx=8)

        self.brush_btn = tk.Button(self.tool_frame, text="Brush", command=self.use_brush)
        self.brush_btn.grid(row=0, column=0, padx=4)
        self.eraser_btn = tk.Button(self.tool_frame, text="Eraser", command=self.use_eraser)
        self.eraser_btn.grid(row=0, column=1, padx=4)
        self.undo_btn = tk.Button(self.tool_frame, text="Undo", command=self.undo)
        self.undo_btn.grid(row=0, column=2, padx=4)
        self.redo_btn = tk.Button(self.tool_frame, text="Redo", command=self.redo)
        self.redo_btn.grid(row=0, column=3, padx=4)
        self.clear_btn = tk.Button(self.tool_frame, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=4, padx=4)

        self.size_label = tk.Label(self.tool_frame, text="Size")
        self.size_label.grid(row=1, column=0, pady=4)
        self.size_slider = tk.Scale(self.tool_frame, from_=1, to=64, orient=tk.HORIZONTAL)
        self.size_slider.set(BRUSH_DEFAULT)
        self.size_slider.grid(row=1, column=1, columnspan=2, sticky="we", padx=2)

        # Model and generate controls
        self.model_frame = tk.Frame(controls_frame)
        self.model_frame.pack(side=tk.LEFT, padx=8)

        self.load_model_btn = tk.Button(self.model_frame, text="Load model", command=self.load_model)
        self.load_model_btn.grid(row=0, column=0, padx=4)
        self.generate_btn = tk.Button(self.model_frame, text="Generate", command=self.generate_once)
        self.generate_btn.grid(row=0, column=1, padx=4)
        self.realtime_var = tk.IntVar(value=1)
        self.realtime_cb = tk.Checkbutton(self.model_frame, text="Real-time preview", variable=self.realtime_var)
        self.realtime_cb.grid(row=0, column=2, padx=4)

        # status
        self.status_label = tk.Label(root, text="No model loaded", anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # internal
        self.tool = "brush"
        self.brush_size = BRUSH_DEFAULT
        self.last_x = None
        self.last_y = None
        self.drawing = False

        # undo/redo stacks store PIL images
        self.history = []
        self.redo_stack = []
        self.push_history()  # initial state

        # model
        self.model = None
        self.model_device = "cpu"
        self.transform_resize = None
        if TORCH_AVAILABLE:
            self.transform_resize = transforms = None  # placeholder if needed

        # preview debounce
        self.preview_after_id = None
        self.preview_running = False

    # -----------------------
    # Drawing methods
    # -----------------------
    def push_history(self):
        if len(self.history) >= HISTORY_LIMIT:
            self.history.pop(0)
        self.history.append(self.image.copy())
        # clearing redo on new action
        self.redo_stack.clear()

    def undo(self):
        if len(self.history) <= 1:
            return
        last = self.history.pop()
        self.redo_stack.append(last)
        self.image = self.history[-1].copy()
        self.draw = ImageDraw.Draw(self.image)
        self._refresh_canvas()
        self.schedule_preview()

    def redo(self):
        if not self.redo_stack:
            return
        img = self.redo_stack.pop()
        self.history.append(img.copy())
        self.image = img.copy()
        self.draw = ImageDraw.Draw(self.image)
        self._refresh_canvas()
        self.schedule_preview()

    def clear(self):
        self.push_history()
        self.image = Image.new("L", (CANVAS_PIX, CANVAS_PIX), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self._refresh_canvas()
        self.schedule_preview()

    def use_brush(self):
        self.tool = "brush"

    def use_eraser(self):
        self.tool = "eraser"

    def on_button_press(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.drawing = True
        self.brush_size = int(self.size_slider.get())
        # start a stroke, save history
        self.push_history()
        self._draw_dot(event.x, event.y)
        self._refresh_canvas()
        self.schedule_preview_debounce()

    def on_paint(self, event):
        if not self.drawing:
            return
        x, y = event.x, event.y
        self.brush_size = int(self.size_slider.get())
        self._draw_line(self.last_x, self.last_y, x, y)
        self.last_x, self.last_y = x, y
        self._refresh_canvas()
        self.schedule_preview_debounce()

    def on_button_release(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        # immediate preview on release if realtime off (or on)
        if not self.realtime_var.get():
            return
        self.schedule_preview()

    def _draw_dot(self, x, y):
        r = self.brush_size // 2
        bbox = [x-r, y-r, x+r, y+r]
        if self.tool == "brush":
            color = 0
        else:
            color = 255
        self.draw.ellipse(bbox, fill=color)

    def _draw_line(self, x0, y0, x1, y1):
        # draw thicker line by drawing many small circles along the path
        distance = max(1, int(((x1-x0)**2 + (y1-y0)**2)**0.5))
        for i in range(distance):
            t = i / max(1, distance)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            self._draw_dot(x, y)

    def _refresh_canvas(self):
        # update PhotoImage on canvas
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_img, image=self.photo)
        self.canvas.update_idletasks()

    # -----------------------
    # Preview scheduling (debounced)
    # -----------------------
    def schedule_preview_debounce(self):
        # called frequently during drawing; debounce updates
        if not self.realtime_var.get():
            return
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(PREVIEW_DEBOUNCE_MS, self.schedule_preview)

    def schedule_preview(self):
        # run preview generation (synchronous in main thread)
        if not self.realtime_var.get() and not self.preview_running:
            return
        # we run model inference in a separate thread to avoid blocking UI
        if self.model is None:
            # fall back to showing a resized sketch
            self._update_preview_from_sketch()
            return
        if self.preview_running:
            return
        self.preview_running = True
        # run inference in a thread (model can be on GPU)
        t = threading.Thread(target=self._run_model_and_update_preview, daemon=True)
        t.start()

    # -----------------------
    # Model loading / inference
    # -----------------------
    def load_model(self):
        if not TORCH_AVAILABLE:
            messagebox.showerror("PyTorch missing", "PyTorch is not installed. Install it with 'pip install torch'.")
            return
        path = filedialog.askopenfilename(title="Select PyTorch model (.pth)", filetypes=[("PyTorch", "*.pth"), ("All files", "*.*")])
        if not path:
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(path, map_location=device)
            # build model instance (should match training architecture)
            model = UNetGenerator(in_channels=1, out_channels=3, base_features=128, depth=4)
            #model = UNetGenerator(in_channels=1, out_channels=3, base_features=64, depth=4)
            try:
                model.load_state_dict(state)
            except RuntimeError as e:
                # try non-strict load if exact mismatch
                try:
                    model.load_state_dict(state, strict=False)
                    messagebox.showwarning("Load checkpoint (non-strict)", "Model loaded with strict=False (some keys ignored). If results look wrong, recreate/adjust the model definition to match training.")
                except Exception as ex:
                    raise ex
            model.eval()
            model.to(device)
            self.model = model
            self.model_device = device
            self.status_label.config(text=f"Model loaded: {os.path.basename(path)} on {device}")
            # do one preview
            self.schedule_preview()
        except Exception as e:
            messagebox.showerror("Model load error", f"Failed to load model:\n{e}")

    def _run_model_and_update_preview(self):
        try:
            # create 128x128 binary sketch input consistent with training: 0..1 float
            sketch = self.image.resize((MODEL_SIZE, MODEL_SIZE), resample=Image.BILINEAR)
            # convert to numpy 0..1, ensure single channel shape 1xHxW
            arr = np.array(sketch).astype(np.float32) / 255.0
            # threshold to binary 0/1 (user requested binary input). Keep as {0,1}
            arr_bin = (arr < 0.5).astype(np.float32)  # black lines -> 1.0
            # invert or not: depending on training (here we send as single channel where 1 means stroke)
            inp = torch.from_numpy(arr_bin).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
            inp = inp.to(self.model_device)
            with torch.no_grad():
                out = self.model(inp)
            # out is tensor 1x3xHxW with values [0,1] usually (if Sigmoid)
            out = out.clamp(0.0, 1.0).cpu().numpy()[0]  # 3xHxW
            out_img = np.transpose(out, (1,2,0))  # HxWx3 in 0..1
            out_img = (out_img * 255).astype(np.uint8)
            pil = Image.fromarray(out_img)
            # resize to preview canvas size
            pil_preview = pil.resize((CANVAS_PIX//2, CANVAS_PIX//2), resample=Image.BILINEAR)
            # update in main thread
            self.root.after(0, self._set_preview_image, pil_preview)
        except Exception as e:
            # on error, show a grayscale resized sketch instead
            print("Model inference error:", e)
            self.root.after(0, self._update_preview_from_sketch)
        finally:
            self.preview_running = False

    def _update_preview_from_sketch(self):
        # show a resized, colorized (simple) version of the sketch as fallback
        sketch = self.image.resize((CANVAS_PIX//2, CANVAS_PIX//2), resample=Image.BILINEAR)
        skew = ImageOps.invert(sketch.convert("L")).convert("RGB")
        self._set_preview_image(skew)

    def _set_preview_image(self, pil_img):
        self.preview_image = pil_img
        self.preview_photo = ImageTk.PhotoImage(self.preview_image)
        self.preview_canvas.itemconfig(self.preview_canvas_img, image=self.preview_photo)
        self.preview_canvas.update_idletasks()

    def generate_once(self):
        # manual generate (synchronous spawn thread)
        if self.model is None:
            self._update_preview_from_sketch()
            return
        self.schedule_preview()

# -----------------------
# main
# -----------------------
def main():
    root = tk.Tk()
    app = SketchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
