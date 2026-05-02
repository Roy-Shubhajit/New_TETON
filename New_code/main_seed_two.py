from __future__ import annotations

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool, global_max_pool

try:
    from torcheeg.datasets.constants import SEED_ADJACENCY_MATRIX
except Exception:
    SEED_ADJACENCY_MATRIX = None


SEED_VII_ROOT = "/hdfs1/Data/Shubhajit/Project/New_TETON/New_code/SEED-VII"
EEG_PREPROCESSED_DIR = Path(SEED_VII_ROOT) / "EEG_preprocessed"
EMOTION_LABEL_FILE = Path(SEED_VII_ROOT) / "emotion_label_and_stimuli_order.xlsx"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

FS = 200
BANDS: Tuple[Tuple[int, int], ...] = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))


def _xlsx_shared_strings(z: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in z.namelist():
        return []
    root = ET.fromstring(z.read(path))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    out: List[str] = []
    for si in root.iter(f"{ns}si"):
        texts = [t.text or "" for t in si.iter(f"{ns}t")]
        out.append("".join(texts))
    return out


def _xlsx_sheet_cells(z: zipfile.ZipFile, sheet_rel_path: str) -> Dict[str, str]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = _xlsx_shared_strings(z)
    root = ET.fromstring(z.read(sheet_rel_path))
    cells: Dict[str, str] = {}
    for c in root.iter(f"{ns}c"):
        ref = c.attrib.get("r", "")
        t = c.attrib.get("t", "")
        v = c.find(f"{ns}v")
        is_node = c.find(f"{ns}is")
        if t == "s" and v is not None and v.text is not None:
            idx = int(v.text)
            cells[ref] = shared[idx] if 0 <= idx < len(shared) else ""
        elif t == "inlineStr" and is_node is not None:
            t_node = is_node.find(f"{ns}t")
            cells[ref] = (t_node.text or "") if t_node is not None else ""
        elif v is not None and v.text is not None:
            cells[ref] = v.text
    return cells


def load_seed_vii_emotion_labels(label_file: Path) -> Tuple[List[int], Dict[str, int]]:
    with zipfile.ZipFile(label_file) as z:
        cells = _xlsx_sheet_cells(z, "xl/worksheets/sheet1.xml")

    session_rows = []
    for ref, value in cells.items():
        m = re.fullmatch(r"A(\d+)", ref)
        if m and value.strip().lower().startswith("session"):
            session_rows.append(int(m.group(1)))
    session_rows = sorted(session_rows)
    if len(session_rows) != 4:
        raise ValueError(f"Expected 4 session rows, found {len(session_rows)} in {label_file}")

    labels_text: List[str] = []
    for row in session_rows:
        for col in "BCDEFGHIJKLMNOPQRSTU":
            val = cells.get(f"{col}{row}", "").strip()
            if val:
                labels_text.append(val)
    if len(labels_text) != 80:
        raise ValueError(f"Expected 80 trial labels, found {len(labels_text)}")

    emo2id: Dict[str, int] = {}
    for e in labels_text:
        if e not in emo2id:
            emo2id[e] = len(emo2id)
    labels = [emo2id[e] for e in labels_text]
    return labels, emo2id


def fallback_ring_adjacency(n_nodes: int = 62) -> np.ndarray:
    a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        a[i, j] = 1
        a[j, i] = 1
    return a


def build_static_edge_index() -> torch.Tensor:
    if SEED_ADJACENCY_MATRIX is not None:
        adj = np.asarray(SEED_ADJACENCY_MATRIX, dtype=np.float32)
    else:
        adj = fallback_ring_adjacency(62)
    rows, cols = np.nonzero(adj)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def build_knn_edge_index(eeg_window: np.ndarray, k: int) -> torch.Tensor:
    # eeg_window: [62, T]
    corr = np.corrcoef(eeg_window)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    score = np.abs(corr)
    n = score.shape[0]
    edges = set()
    for i in range(n):
        nn = np.argpartition(-score[i], kth=min(k, n - 1) - 1)[:k]
        for j in nn:
            if i == j:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    rows, cols = [], []
    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def window_to_features(eeg_window: np.ndarray) -> torch.Tensor:
    # eeg_window: [62, W] -> x: [62,5]
    freqs, psd = welch(eeg_window, fs=FS, nperseg=min(FS, eeg_window.shape[1]), axis=1)
    feats = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        feats.append(np.log(power + 1e-8))
    return torch.tensor(np.stack(feats, axis=1), dtype=torch.float32)


def build_seed_graphs(
    eeg_dir: Path,
    labels_80: List[int],
    graph_method: str,
    knn_k: int,
    window_size: int,
    window_stride: int,
    max_subjects: Optional[int],
    cache_tag: str,
    use_cache: bool,
) -> List[Data]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"seed_vii_{cache_tag}.pt"
    if use_cache and cache_path.exists():
        try:
            bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            return bundle["graphs"]
        except Exception as e:
            print(f"[WARN] Failed to read cache {cache_path}: {e}")
            print("[WARN] Cache appears corrupted. Deleting and rebuilding...")
            try:
                cache_path.unlink()
            except Exception as del_err:
                print(f"[WARN] Could not delete corrupted cache: {del_err}")

    static_edge = build_static_edge_index()
    graphs: List[Data] = []

    files = sorted(eeg_dir.glob("*.mat"), key=lambda p: int(p.stem))
    if max_subjects is not None:
        files = files[: max(1, max_subjects)]

    for mat_path in tqdm(files):
        subject_id = int(mat_path.stem)
        mat = loadmat(mat_path)
        for trial_id in range(1, 81):
            key = str(trial_id)
            if key not in mat:
                continue
            eeg_trial = np.asarray(mat[key], dtype=np.float32)  # [62, T]
            T = eeg_trial.shape[1]
            if T < window_size:
                continue

            for start in range(0, T - window_size + 1, window_stride):
                w = eeg_trial[:, start : start + window_size]
                x = window_to_features(w)
                #print(x.shape)
                if graph_method == "knn_corr":
                    edge_index = build_knn_edge_index(w, k=knn_k)
                else:
                    edge_index = static_edge
                y = labels_80[trial_id - 1]
                graphs.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        y=torch.tensor([y], dtype=torch.long),
                        subject=torch.tensor([subject_id], dtype=torch.long),
                        trial=torch.tensor([trial_id], dtype=torch.long),
                    )
                )

    if not graphs:
        raise ValueError("No graphs were created. Try smaller --window_sec or lower --max_subjects debug settings.")

    if use_cache:
        try:
            torch.save({"graphs": graphs}, cache_path)
            print(f"Cached graphs to: {cache_path}")
        except Exception as e:
            # Do not fail training if cache write fails (disk quota/space/permissions).
            print(f"[WARN] Cache write failed at {cache_path}: {e}")
            print("[WARN] Continuing without cache.")
    return graphs


def normalize_graph_features(train_ds: List[Data], val_ds: List[Data], test_ds: List[Data]) -> None:
    train_x = torch.cat([d.x for d in train_ds], dim=0)
    mu = train_x.mean(dim=0, keepdim=True)
    sigma = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    for ds in (train_ds, val_ds, test_ds):
        for d in ds:
            d.x = (d.x - mu) / sigma


class GCN(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = torch.nn.ModuleList([GCNConv(in_features, hidden)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data: Data):
        h = data.x
        for conv in self.convs:
            h = conv(h, data.edge_index).relu()
        h = global_mean_pool(h, data.batch)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


class GIN(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(Sequential(Linear(in_features, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden), ReLU()))
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden), ReLU()))
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data: Data):
        h = data.x
        for conv in self.convs:
            h = conv(h, data.edge_index)
        h = global_add_pool(h, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


def make_model(model_name: str, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
    if model_name.lower() == "gcn":
        return GCN(in_features, num_classes, hidden, dropout, num_layers)
    return GIN(in_features, num_classes, hidden, dropout, num_layers)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, total_correct, total, n_batches = 0.0, 0, 0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=use_amp)
        with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
            logits = model(batch)
            y = batch.y.view(-1).long()
            loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += y.numel()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1), total_correct / max(total, 1)


def train_once(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
):
    labels = [int(d.y.item()) for d in train_loader.dataset]
    counts = np.bincount(labels)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights * (len(counts) / weights.sum())
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val, best_state, stale = -1.0, None, 0
    for epoch in range(epochs + 1):
        model.train()
        total_loss, total_correct, total, n_batches = 0.0, 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=use_amp)
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                logits = model(batch)
                y = batch.y.view(-1).long()
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)
            total_correct += int((pred == y).sum().item())
            total += y.numel()
            total_loss += loss.item()
            n_batches += 1

        tr_loss = total_loss / max(n_batches, 1)
        tr_acc = total_correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:>4} | train_loss {tr_loss:.4f} | train_acc {tr_acc*100:5.2f}% "
                f"| val_loss {val_loss:.4f} | val_acc {val_acc*100:5.2f}%"
            )
        if stale >= patience:
            print(f"Early stop at epoch {epoch}, no val improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    return best_val, test_loss, test_acc


def parse_args():
    p = argparse.ArgumentParser(description="SEED-VII GCN/GIN with KNN graph + Optuna tuning")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--model", type=str, choices=["gcn", "gin"], default="gin")
    p.add_argument("--graph_method", type=str, choices=["seed_static", "knn_corr"], default="seed_static")
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--window_sec", type=float, default=8.0)
    p.add_argument("--stride_sec", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--use_optuna", action="store_true")
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--optuna_epochs", type=int, default=120)
    p.add_argument("--no_cache", action="store_true", help="Disable graph cache read/write.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels_80, emo_map = load_seed_vii_emotion_labels(EMOTION_LABEL_FILE)
    print(f"Loaded labels for 80 trials. Emotion map: {emo_map}")

    win = int(FS * args.window_sec)
    stride = int(FS * args.stride_sec)
    cache_tag = f"raw_{args.graph_method}_k{args.knn_k}_w{win}_s{stride}_sub{args.max_subjects or 'all'}"
    dataset = build_seed_graphs(
        EEG_PREPROCESSED_DIR,
        labels_80,
        args.graph_method,
        args.knn_k,
        win,
        stride,
        args.max_subjects,
        cache_tag,
        use_cache=not args.no_cache,
    )
    print(f"Graphs: {len(dataset)}")

    y_all = [int(d.y.item()) for d in dataset]
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y_all)
    y_train_tmp = [y_all[i] for i in train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, stratify=y_train_tmp)
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]
    normalize_graph_features(train_ds, val_ds, test_ds)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable; using CPU.")

    num_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda, "persistent_workers": num_workers > 0}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    in_features = train_ds[0].num_node_features
    n_classes = len(emo_map)

    if args.use_optuna:
        try:
            import optuna
        except Exception as e:
            raise RuntimeError(
                "Optuna is not installed. Install it with: pip install optuna"
            ) from e

        def objective(trial):
            model_name = trial.suggest_categorical("model", ["gcn", "gin"])
            hidden = trial.suggest_int("hidden", 64, 256, step=32)
            num_layers = trial.suggest_int("num_layers", 2, 6)
            dropout = trial.suggest_float("dropout", 0.1, 0.6)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [64, 128, 256])

            tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
            va_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)
            te_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs)

            model = make_model(model_name, in_features, n_classes, hidden, dropout, num_layers).to(device)
            val_acc, _, _ = train_once(model, tr_loader, va_loader, te_loader, device, args.optuna_epochs, lr, wd, 40)
            return val_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)
        bp = study.best_trial.params
        print(f"Optuna best params: {bp}, best val={study.best_value*100:.2f}%")
        args.model = bp["model"]
        args.hidden = int(bp["hidden"])
        args.num_layers = int(bp["num_layers"])
        args.dropout = float(bp["dropout"])
        args.lr = float(bp["lr"])
        args.weight_decay = float(bp["weight_decay"])
        args.batch_size = int(bp["batch_size"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = make_model(args.model, in_features, n_classes, args.hidden, args.dropout, args.num_layers).to(device)
    print(
        f"Final training: model={args.model}, graph={args.graph_method}, k={args.knn_k}, "
        f"epochs={args.epochs}, batch={args.batch_size}, hidden={args.hidden}, layers={args.num_layers}, "
        f"lr={args.lr:.2e}, wd={args.weight_decay:.2e}"
    )
    best_val, test_loss, test_acc = train_once(
        model, train_loader, val_loader, test_loader, device, args.epochs, args.lr, args.weight_decay, args.patience
    )
    print(f"Best Val Acc: {best_val*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
