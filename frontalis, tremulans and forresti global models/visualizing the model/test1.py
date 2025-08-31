#!/usr/bin/env python3
# torch>=1.7  torchview>=0.2.6  graphviz system binaries required
import os
import torch, torch.nn as nn
from torchview import draw_graph

class LargerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 10×10→5×5
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,2)
        )
    def forward(self,x): return self.classifier(self.features(x))

OUT = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/frontalis, tremulans and forresti global models/visualizing the model/store"          # <-- choose any existing folder
os.makedirs(OUT, exist_ok=True)

# 1️⃣ DON’T give Torchview file args yet – just get the graph object
cg = draw_graph(
        LargerCNN(),
        input_size=(1,1,10,10),
        expand_nested=True,        # show every submodule
        # ⬆ no directory / filename here
)

# 2️⃣ Optionally resize everything for readability
cg.resize_graph(scale=2.0)        # 2× default font & node sizes

# 3️⃣ Now export in any format you like
gv = cg.visual_graph              # the underlying graphviz.Digraph
gv.graph_attr.update(dpi="300")   # hi-res raster
gv.render(directory=OUT, filename="larger_cnn_300dpi", format="png", cleanup=True)
gv.render(directory=OUT, filename="larger_cnn_vector",  format="pdf", cleanup=True)

print("✓ diagrams saved in", OUT)
