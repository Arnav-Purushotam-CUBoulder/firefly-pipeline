import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# Read in data from folder dir string
def load_data(dir_name):
    data = []
    for file_name in os.listdir(dir_name):
        if file_name.endswith(".png"):
            file_path = os.path.join(dir_name, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            data.append(image)
    return np.array(data)

# Apply transform function to list of patches
def transform(firefly, background):
    # Combine all patches from firefly and background
    all_patches = np.concatenate((firefly, background), axis=0)

    # Calculate mean and std for each channel across all patches
    target_mean = np.mean(all_patches, axis=(0, 1, 2))
    target_std = np.std(all_patches, axis=(0, 1, 2))

    # Normalize all patches to the global average
    epsilon = 1e-8
    firefly_transform = (firefly - target_mean) / (target_std + epsilon)
    background_transform = (background - target_mean) / (target_std + epsilon)

    return np.array(firefly_transform), np.array(background_transform)



def learn_model(train_transform):
    import numpy as _np
    from tqdm import tqdm
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    # Unpack into X (N,C,H,W) and y (N,)
    if isinstance(train_transform, tuple) and len(train_transform) == 2:
        X, y = train_transform
        X = _np.asarray(X)
        y = _np.asarray(y).astype(_np.int64)
        if X.ndim != 4:
            raise ValueError("X_train must have shape (N,H,W,C) or (N,C,H,W)")
        X_nchw = X if X.shape[1] in (1,3) else _np.transpose(X, (0,3,1,2))
    elif isinstance(train_transform, dict):
        fire = _np.asarray(train_transform.get('firefly'))
        back = _np.asarray(train_transform.get('background'))
        if fire is None or back is None:
            raise ValueError("Dict must have 'firefly' and 'background' arrays")
        X = _np.concatenate([fire, back], axis=0)
        y = _np.concatenate([
            _np.ones(len(fire), dtype=_np.int64),
            _np.zeros(len(back), dtype=_np.int64)
        ], axis=0)
        if X.ndim != 4:
            raise ValueError("Input arrays must have shape (N,H,W,C) or (N,C,H,W)")
        X_nchw = X if X.shape[1] in (1,3) else _np.transpose(X, (0,3,1,2))
    else:
        raise ValueError("train_transform must be a (X,y) tuple or a dict with 'firefly'/'background'")

    X_t = torch.from_numpy(X_nchw).float()
    y_t = torch.from_numpy(y).long()

    DEVICE = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    # Weighted sampler for class imbalance
    import numpy as _np2
    class_counts = _np2.bincount(y)
    total = class_counts.sum()
    weights = torch.tensor([total / max(c, 1) for c in class_counts], dtype=torch.float)
    sample_weights = weights[y_t]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_t), replacement=True)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=128, sampler=sampler, shuffle=False)

    # Small CNN appropriate for tiny patches
    class TinyCNN(nn.Module):
        def __init__(self, in_ch, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(32, num_classes)
        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.fc(x)

    in_ch = X_t.shape[1]
    model = TinyCNN(in_ch).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for ep in range(1, 21):
        model.train();
        for xb, yb in tqdm(dl, desc=f"train {ep:02d}", ncols=100, leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step()
    model.eval()
    return model



def evaluate_model(model, test_transform):
    import numpy as _np
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if isinstance(test_transform, tuple) and len(test_transform) == 2:
        X, y = test_transform
        X = _np.asarray(X)
        y = _np.asarray(y).astype(_np.int64)
        if X.ndim != 4:
            raise ValueError("X_test must have shape (N,H,W,C) or (N,C,H,W)")
        X_nchw = X if X.shape[1] in (1,3) else _np.transpose(X, (0,3,1,2))
    elif isinstance(test_transform, dict):
        fire = _np.asarray(test_transform.get('firefly'))
        back = _np.asarray(test_transform.get('background'))
        if fire is None or back is None:
            raise ValueError("Dict must have 'firefly' and 'background' arrays")
        X = _np.concatenate([fire, back], axis=0)
        y = _np.concatenate([
            _np.ones(len(fire), dtype=_np.int64),
            _np.zeros(len(back), dtype=_np.int64)
        ], axis=0)
        if X.ndim != 4:
            raise ValueError("Input arrays must have shape (N,H,W,C) or (N,C,H,W)")
        X_nchw = X if X.shape[1] in (1,3) else _np.transpose(X, (0,3,1,2))
    else:
        raise ValueError("test_transform must be a (X,y) tuple or a dict with 'firefly'/'background'")

    X_t = torch.from_numpy(X_nchw).float()
    y_t = torch.from_numpy(y).long()
    dl = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=False)

    DEVICE = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE).eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = corr = tot = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            test_loss += criterion(out, yb).item()*xb.size(0)
            corr += (out.argmax(1)==yb).sum().item(); tot += xb.size(0)

    acc = (corr / tot) if tot else 0.0
    return {"test_loss": (test_loss / tot) if tot else 0.0,
            "test_acc": acc,
            "n_samples": tot}


def main():
    firefly = load_data('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/standardization function data/frontalis and pyrallis balanced data/pyralis_firefly')
    background = load_data('/Users/arnavps/Desktop/New DL project data to transfer to external disk/pyrallis related data/standardization function data/frontalis and pyrallis balanced data/pyralis_background')

    firefly_transform, background_transform = transform(firefly, background) 

    # Arnav - edit below:

    # Split data:
    firefly_train, firefly_test = train_test_split(
        firefly_transform, test_size=0.25, random_state=1
    )
    background_train, background_test = train_test_split(
        background_transform, test_size=0.25, random_state=1
    )
    
    train_transform = {
        'firefly': firefly_train,
        'background': background_train,
    }
    test_transform = {
        'firefly': firefly_test,
        'background': background_test,
    }

    model = learn_model(train_transform)

    results = evaluate_model(model, test_transform)
    print(results)

if __name__ == "__main__":
    main()
