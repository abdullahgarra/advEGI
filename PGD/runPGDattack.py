# Experiment 2: role-identification on airport dataset


import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from models.dgi import DGI, MultiClassifier
from models.subgi_pgd import SubGI
from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding
import random
from typing import List, Tuple
from contextlib import contextmanager

import os, csv
from datetime import datetime


def createTraining_seeded(labels, seed, train_ratio=0.8):
    rs = np.random.RandomState(seed)
    n = labels.shape[0]
    idx = np.arange(n); rs.shuffle(idx)
    num_train = int(n * train_ratio)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:num_train]] = True
    test_mask[idx[num_train:]]  = True
    return train_mask, test_mask


def _jsd(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def _kl(a,b): 
        a = np.clip(a, eps, 1); b = np.clip(b, eps, 1); 
        return (a * (np.log(a) - np.log(b))).sum()
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def degree_hist_jsd(A0: np.ndarray, A1: np.ndarray) -> float:
    d0 = A0.sum(1); d1 = A1.sum(1)
    maxd = int(max(d0.max(), d1.max()))
    h0 = np.bincount(d0.astype(int), minlength=maxd+1)
    h1 = np.bincount(d1.astype(int), minlength=maxd+1)
    return float(_jsd(h0, h1))


def eval_transfer_once(encoder_model, target_name: str, args, head_epochs: int = 50):
    # load target graph
    fpath, lpath = guess_ds_paths(target_name)
    nxg, tlabels = read_struct_net(argparse.Namespace(file_path=fpath, label_path=lpath))
    tg, tlabels = constructDGL(nxg, tlabels)
    tg.readonly()
    tlabels = torch.LongTensor(tlabels)
    tfeatures = _build_features(tg, args)

    # train/test split on target (same ratio as source)
    seed_map = {"usa": 10_001, "brazil": 10_002}  # any constants
    t_train_mask, t_test_mask = createTraining_seeded(tlabels, seed=seed_map[target_name.lower()])
    #t_train_mask, t_test_mask = createTraining(tlabels, None)

    # get embeddings from current encoder
    with swap_model_graph(encoder_model, tg):
        Z = encoder_model.encoder(tfeatures, corrupt=False).detach()

    # train a small linear head (constant recipe)
    clf = MultiClassifier(args.n_hidden, int(tlabels.max().item()) + 1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay)
    for _ in range(head_epochs):
        clf.train()
        opt.zero_grad()
        preds = clf(Z)
        loss = F.nll_loss(preds[t_train_mask], tlabels[t_train_mask])
        loss.backward(); opt.step()

    acc = evaluate(clf, Z, tlabels, t_test_mask)
    return float(acc)


def guess_ds_paths(name: str):
    name = name.lower()
    if name in ["eu","europe","europe-airports","european"]:
        return ("data/europe-airports.edgelist", "data/labels-europe-airports.txt")
    if name in ["usa","us","united-states"]:
        return ("data/usa-airports.edgelist", "data/labels-usa-airports.txt")
    if name in ["brazil","br"]:
        return ("data/brazil-airports.edgelist", "data/labels-brazil-airports.txt")
    raise ValueError(f"Unknown dataset alias: {name}")


def _build_features(g: dgl.DGLGraph, args):
    # degree-bucket one-hot of size n_hidden
    return degree_bucketing(g, args, None)


@contextmanager
def swap_model_graph(model, g_new):
    """Temporarily point every submodule that caches a graph to g_new."""
    # Save old refs
    old_main = getattr(model, 'g', None)

    old_enc = getattr(getattr(model, 'encoder', None), 'g', None)
    old_enc_conv_g = None
    if getattr(model, 'encoder', None) is not None and hasattr(model.encoder, 'conv'):
        old_enc_conv_g = getattr(model.encoder.conv, 'g', None)

    old_disc = getattr(getattr(model, 'subg_disc', None), 'g', None)

    try:
        # Swap root
        if hasattr(model, 'g'):
            model.g = g_new

        # Swap encoder and its inner conv
        if getattr(model, 'encoder', None) is not None:
            if hasattr(model.encoder, 'g'):
                model.encoder.g = g_new
            if hasattr(model.encoder, 'conv') and hasattr(model.encoder.conv, 'g'):
                model.encoder.conv.g = g_new

        # Swap discriminator
        if getattr(model, 'subg_disc', None) is not None and hasattr(model.subg_disc, 'g'):
            model.subg_disc.g = g_new

        yield

    finally:
        # Restore root
        if hasattr(model, 'g'):
            model.g = old_main

        # Restore encoder + conv
        if getattr(model, 'encoder', None) is not None:
            if hasattr(model.encoder, 'g'):
                model.encoder.g = old_enc
            if hasattr(model.encoder, 'conv') and hasattr(model.encoder.conv, 'g'):
                model.encoder.conv.g = old_enc_conv_g

        # Restore discriminator
        if getattr(model, 'subg_disc', None) is not None and hasattr(model.subg_disc, 'g'):
            model.subg_disc.g = old_disc



def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def spectral_feature(graph, args):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        #OUT.write('0 {} {} 1\n'.format(id_a, id_b))
        A[id_a, id_b] = 1
    embedding = SpectralEmbedding(n_components=args.n_hidden)
    features = torch.FloatTensor(embedding.fit_transform(A))
    return features

def degree_bucketing(graph, args, degree_emb=None, max_degree = 10):
    #G = nx.DiGraph(graph)
    #embed()
    max_degree = args.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    #return features
    # embed()
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
            # features[i, :] = degree_emb[min(graph.degree(i), max_degree-1), :]
        except:
            features[i][0] = 1
            #features[i, :] = degree_emb[0, :]
    # embed()
    #embed()
    return features

def createTraining(labels, valid_mask = None, train_ratio=0.8):
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    #for i in range(len(idx) * train_ratio):
    # embed()
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask

def read_struct_net(args):
    #g = DGLGraph()
    g = nx.Graph()
    #g.add_nodes(1000)
    with open(args.file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            # print(tmp[0], tmp[1])
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(args.label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels
    
def constructDGL(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    # embed()
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
    
    # embed()
    return new_g, relabels

def output_adj(graph):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1
    return A

def graph_from_adj_numpy(A: np.ndarray) -> dgl.DGLGraph:
    """Build an undirected DGLGraph from a 0/1 numpy adjacency (zero diag, symmetric)."""
    g_new = DGLGraph()
    n = A.shape[0]
    g_new.add_nodes(n)
    # keep self-loops separate from adversarial flips
    src, dst = np.nonzero(A)
    for u, v in zip(src, dst):
        if u != v:
            g_new.add_edge(int(u), int(v))
    return g_new

def to_numpy_adj_undirected(g: dgl.DGLGraph) -> np.ndarray:
    """0/1 adjacency (no self loops), symmetric."""
    n = g.number_of_nodes()
    A = np.zeros((n, n), dtype=np.uint8)
    a, b = g.all_edges()
    for u, v in zip(a.numpy().tolist(), b.numpy().tolist()):
        if u != v:
            A[u, v] = 1
    # enforce symmetry
    A = ((A + A.T) > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A


def build_candidates(A: np.ndarray, max_nonedge_per_node: int = 10) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Returns (edge_candidates, nonedge_candidates).
    - edge_candidates: existing undirected edges (u<v) — allows deletions
    - nonedge_candidates: 2-hop non-edges per node, top-limited per node — allows additions
    """
    n = A.shape[0]
    # existing edges (upper triangle)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]

    # 2-hop neighbors via A^2 (boolean)
    A2 = (A @ A) > 0
    nonedges = []
    for i in range(n):
        cnt = 0
        for j in range(i+1, n):
            if A[i, j] == 0 and A2[i, j]:
                nonedges.append((i, j))
                cnt += 1
                if cnt >= max_nonedge_per_node:
                    break
    # dedup already ensured by i<j loop
    return edges, nonedges


@torch.no_grad()
def probe_loss_subgi_once(
    model,
    A_tmp,
    features,
    args,
    verbose_label=None,           # str or None
    cand_uv=None,                  # tuple (u, v) or None (for coverage check)
    rng_seed=None                  # int or None (to stabilize negatives sampling)
):
    # --- stable RNG for fair comparisons ---
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if rng_seed is not None:
        torch.manual_seed(int(rng_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rng_seed))

    # --- build temp graph and sample one ego-batch ---
    g_tmp = graph_from_adj_numpy(A_tmp)
    g_tmp.readonly()
    g_tmp.ndata['features'] = features

    sampler = dgl.contrib.sampling.NeighborSampler(
        g_tmp, 256, 5, neighbor_type='in', num_workers=1,
        add_self_loop=False, num_hops=args.n_layers + 1, shuffle=False
    )

    # --- eval mode during probe to reduce noise ---
    was_training = model.training
    model.eval()

    loss_val = 0.0
    with swap_model_graph(model, g_tmp):
        for nf in sampler:
            nf.copy_from_parent()
            loss_val = model(features, nf).item()
            
            # Optional: coverage check (did we touch u or v anywhere in the NodeFlow?)
            coverage = ""
            if cand_uv is not None:
                u, v = cand_uv
                try:
                    # collect all parent IDs present in layers (0..num_blocks), plus seeds (-1)
                    all_layers = [nf.layer_parent_nid(i) for i in range(nf.num_blocks + 1)]
                    all_layers.append(nf.layer_parent_nid(-1))
                    present = torch.cat(all_layers).unique().tolist()
                    hit = (int(u) in present) or (int(v) in present)
                    coverage = f" (coverage: {'hit' if hit else 'miss'})"
                except Exception:
                    coverage = ""
            if verbose_label is not None:
                nnz = int(A_tmp.sum() // 2)
                seeds = int(nf.layer_parent_nid(-1).shape[0])
                print(f"[probe] {verbose_label} | edges={nnz} seeds={seeds} loss={loss_val:.6f}{coverage}")
            break  # exactly one small ego-batch

    # --- restore mode & RNG ---
    if was_training:
        model.train()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)

    return loss_val



def pgd_attack_graph_subgi(
    model: SubGI,
    A_base: np.ndarray,
    features: torch.Tensor,
    args,
    steps: int = 10,
    add_per_step: int = 10,
    del_per_step: int = 10,
    degree_cap: int = 4,
    rand_subset: int = 200,
) -> np.ndarray:
    """
    Black-box PGD on graph structure for SubGI:
    - At each step: evaluate marginal loss gain for a random subset of candidate additions and deletions,
      flip the top-k that increase loss the most (respect symmetry & degree caps).
    - Budget is implicit: steps * (add_per_step + del_per_step) flips max.
    """
    A = A_base.copy().astype(np.uint8)
    n = A.shape[0]
    deg = A.sum(1)
    step_stats = []
    edge_cand, nonedge_cand = build_candidates(A, max_nonedge_per_node=10)

    def degree_ok(u, v, add: bool) -> bool:
        if add:
            return (deg[u] + 1 - (A[u, v]==1) <= deg[u] + degree_cap) and (deg[v] + 1 - (A[u, v]==1) <= deg[v] + degree_cap)
        else:
            return (deg[u] - 1 + (A[u, v]==0) >= max(0, deg[u] - degree_cap)) and (deg[v] - 1 + (A[u, v]==0) >= max(0, deg[v] - degree_cap))

    # ⭐ baseline once before the loop
    base_loss = probe_loss_subgi_once(model, A, features, args, verbose_label="baseline", rng_seed=0)
    probe_seed = 1337
    
    for t in range(steps):
        # sample subsets to keep it quick
        del_pool = random.sample(edge_cand, min(len(edge_cand), rand_subset)) if edge_cand else []
        add_pool = random.sample(nonedge_cand, min(len(nonedge_cand), rand_subset)) if nonedge_cand else []

        # ⭐ step header
        print(f"[pgd] step {t+1}/{steps}: base={base_loss:.6f} | cand_del={len(del_pool)} cand_add={len(add_pool)}")

        gains_del, gains_add = [], []
        show_del = show_add = 0  # only print first few to avoid spam
        SHOW_K = 3

        # --- deletions ---
        for (u, v) in del_pool:
            if A[u, v] == 1 and degree_ok(u, v, add=False):
                A[u, v] = A[v, u] = 0
                deg[u] -= 1; deg[v] -= 1
                # ⭐ deterministic seed per candidate & step
                seed = (t << 21) ^ (u << 10) ^ v
                loss_new = probe_loss_subgi_once(model, A, features, args,
                                                 verbose_label=None, cand_uv=(u, v), rng_seed=probe_seed)
                delta = loss_new - base_loss
                gains_del.append(((u, v), delta))
                if show_del < SHOW_K:
                    print(f"[pgd]   DEL({u},{v}) Δ={delta:+.6f}")
                    show_del += 1
                # revert
                A[u, v] = A[v, u] = 1
                deg[u] += 1; deg[v] += 1

        # --- additions ---
        for (u, v) in add_pool:
            if A[u, v] == 0 and degree_ok(u, v, add=True):
                A[u, v] = A[v, u] = 1
                deg[u] += 1; deg[v] += 1
                seed = (t << 21) ^ (u << 10) ^ v ^ 0x9E3779B9  # different salt from deletions
                loss_new = probe_loss_subgi_once(model, A, features, args,
                                                 verbose_label=None, cand_uv=(u, v), rng_seed=probe_seed)
                delta = loss_new - base_loss
                gains_add.append(((u, v), delta))
                if show_add < SHOW_K:
                    print(f"[pgd]   ADD({u},{v}) Δ={delta:+.6f}")
                    show_add += 1
                # revert
                A[u, v] = A[v, u] = 0
                deg[u] -= 1; deg[v] -= 1

        # sort by gain and pick top-k per type
        gains_del.sort(key=lambda x: x[1], reverse=True)
        gains_add.sort(key=lambda x: x[1], reverse=True)

        # ⭐ print best gains seen this step
        best_del = gains_del[0][1] if gains_del else float('nan')
        best_add = gains_add[0][1] if gains_add else float('nan')
        print(f"[pgd]   bestΔ_del={best_del:+.6f} bestΔ_add={best_add:+.6f}")
        
        # keep only strictly positive gains
        pos_del = [(uv, d) for (uv, d) in gains_del if d > 0]
        pos_add = [(uv, d) for (uv, d) in gains_add if d > 0]

        picked_del = gains_del[:del_per_step]
        picked_add = gains_add[:add_per_step]
        
        if not picked_del or not picked_add:
            print("[pgd]   no positive-gain flips; early stop")
            break

        # apply flips
        applied_del = applied_add = 0
        any_change = False
        for (u, v), g in picked_del:
            if A[u, v] == 1 and degree_ok(u, v, add=False):
                A[u, v] = A[v, u] = 0
                deg[u] -= 1; deg[v] -= 1
                applied_del += 1
                any_change = True
        for (u, v), g in picked_add:
            if A[u, v] == 0 and degree_ok(u, v, add=True):
                A[u, v] = A[v, u] = 1
                deg[u] += 1; deg[v] += 1
                applied_add += 1
                any_change = True

        if any_change:
            prev = base_loss
            base_loss = probe_loss_subgi_once(model, A, features, args,
                                              verbose_label="after-apply", rng_seed=probe_seed)#(t+1))
            # ⭐ applied summary + baseline movement
            print(f"[pgd]   applied +{applied_add} / -{applied_del} | baseline {prev:.6f} → {base_loss:.6f}")
        else:
            print("[pgd]   no positive-gain flips; early stop")
            break
        
        # refresh candidate lists
        step_stats.append({
            "base_before": float(prev),
            "best_delta_add": float(best_add) if not np.isnan(best_add) else 0.0,
            "best_delta_del": float(best_del) if not np.isnan(best_del) else 0.0,
            "applied_add": int(applied_add),
            "applied_del": int(applied_del),
            "base_after": float(base_loss),
        })

        edge_cand, nonedge_cand = build_candidates(A, max_nonedge_per_node=10)

    print(f"[pgd] end | edges={int(A.sum()//2)} final_base={base_loss:.6f}")
    return A, step_stats

# dump the best run
def main(args):
    import os, dgl
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # If you ever run on GPU:
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # DGL has its own RNG:
    try:
        dgl.random.seed(args.seed)   # older DGL
    except AttributeError:
        dgl.seed(args.seed)          # newer DGL

    # Deterministic PyTorch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess dataset
    #data = load_data(args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)
    src_name = ("europe" if "europe" in args.file_path.lower() else
                "usa"    if "usa"    in args.file_path.lower() else
                "brazil" if "brazil" in args.file_path.lower() else "source")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.logdir, f"pgd_curve_{src_name}_{stamp}.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    targets = [t.strip() for t in args.eval_targets.split(",") if t.strip()]
    header = ["epoch","pgd_base_before","pgd_base_after","delta_base",
            "applied_add","applied_del","deg_hist_jsd"] + [f"acc_{t}" for t in targets]
    csv_w.writerow(header); csv_f.flush()

    if False:
        graphs = create(args)
        max_num, max_id = 0,-1
        for idx, g in enumerate(graphs):
            if g.number_of_edges() > max_num:
                max_num = g.number_of_edges()
                max_id = idx
    torch.manual_seed(2)
    #embed()
    test_acc = []
    for runs in tqdm(range(10)):
        g,labels = read_struct_net(args)
        valid_mask = None
        if True:
            g.remove_edges_from(nx.selfloop_edges(g))

        g, labels = constructDGL(g, labels)

        labels = torch.LongTensor(labels)
        
        degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, args.n_hidden])), requires_grad=False)

        #features = torch.FloatTensor(data.features)
        if True:
            features = degree_bucketing(g, args, degree_emb)
        else:
            features = spectral_feature(g, args)
        # embed()
        #features = torch.FloatTensor(np.random.normal(0, 1, [graph.number_of_nodes(), args.n_hidden]))
        

        train_mask, test_mask = createTraining(labels, valid_mask)
        # labels = torch.LongTensor(labels)
        # embed()
        if True:
            if hasattr(torch, 'BoolTensor'):
                train_mask = torch.BoolTensor(train_mask)
                #val_mask = torch.BoolTensor(val_mask)
                test_mask = torch.BoolTensor(test_mask)
            else:
                train_mask = torch.ByteTensor(train_mask)
                #val_mask = torch.ByteTensor(val_mask)
                test_mask = torch.ByteTensor(test_mask)
        # embed()
        in_feats = features.shape[1]
        n_classes = labels.max().item() + 1
        n_edges = g.number_of_edges()

        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            torch.cuda.set_device(args.gpu)
            features = features.cuda()
            labels = labels.cuda()

        

        g.readonly()
        n_edges = g.number_of_edges()

        # create DGI model
        if args.model_type == 1:
            dgi = VGAE(g,
                in_feats,
                args.n_hidden,
                args.n_hidden,
                #F.relu,
                args.dropout)
            #dgi = DGI(g,
            #        in_feats,
            #        args.n_hidden,
            #        args.n_layers,
            #        nn.PReLU(args.n_hidden),
            #        args.dropout)
            dgi.prepare()
            #embed()
            #dgi.adj_train = g.adjacency_matrix_scipy()
            dgi.adj_train = sp.csr_matrix(output_adj(g))
            # embed()

        elif args.model_type == 0:
            dgi = DGI(g,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout)
        elif args.model_type == 2:
            dgi = SubGI(g,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout,
                    args.model_id)
        # print(dgi)
        if cuda:
            dgi.cuda()

        dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                        lr=args.dgi_lr,
                                        weight_decay=args.weight_decay)

        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        g.ndata['features'] = features
        for epoch in range(args.n_dgi_epochs):
            train_sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,  # 0,
                                                                    neighbor_type='in', num_workers=1,
                                                                    add_self_loop=False,
                                                                    num_hops=args.n_layers + 1, shuffle=True)
            dgi.train()
            if epoch >= 3:
                t0 = time.time()
            
            loss = 0.0
            # VGAE mode
            if args.model_type == 1:
                dgi.optimizer = dgi_optimizer
                dgi.train_sampler = train_sampler
                dgi.features = features
                loss = dgi.train_model()
            # EGI mode
            elif args.model_type == 2:
                print(f"PGD attack step, epoch {epoch}", flush=True)
                A_base = to_numpy_adj_undirected(g)
                pgd_base_before = None; pgd_base_after = None
                applied_add = applied_del = 0
                deg_jsd = 0.0

                if args.pgd_steps > 0:
                    A_adv, step_stats = pgd_attack_graph_subgi(
                        model=dgi, A_base=A_base, features=features, args=args,
                        steps=args.pgd_steps, add_per_step=args.pgd_add_k,
                        del_per_step=args.pgd_del_k, degree_cap=args.pgd_degree_cap,
                        rand_subset=args.pgd_rand_subset
                    )
                    if step_stats:
                        pgd_base_before = step_stats[0]["base_before"]
                        pgd_base_after  = step_stats[-1]["base_after"]
                        applied_add = sum(s["applied_add"] for s in step_stats)
                        applied_del = sum(s["applied_del"] for s in step_stats)
                    else:
                        pgd_base_before = pgd_base_after = probe_loss_subgi_once(dgi, A_base, features, args, verbose_label="baseline", rng_seed=1337)

                    deg_jsd = degree_hist_jsd(A_base, A_adv)
                    if args.features_recompute_per_epoch:
                        features = degree_bucketing(graph_from_adj_numpy(A_adv), args, None)  # recompute degree-buckets on adv graph

                    g_train = graph_from_adj_numpy(A_adv)
                else:
                    # clean training
                    g_train = g
                    pgd_base_before = pgd_base_after = probe_loss_subgi_once(dgi, A_tmp=A_base, features=features, args=args, verbose_label="baseline", rng_seed=1337)
                    deg_jsd = 0.0

                g_train.readonly()
                g_train.ndata['features'] = features
                train_sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train, 256, 5, neighbor_type='in', num_workers=1,
                    add_self_loop=False, num_hops=args.n_layers + 1, shuffle=True
                )

                with swap_model_graph(dgi, g_train):
                    for nf in train_sampler:
                        dgi_optimizer.zero_grad()
                        l = dgi(features, nf)
                        l.backward()
                        loss += l
                        dgi_optimizer.step()

                # === Eval on targets (epoch-wise) ===
                row = [epoch,
                    float(pgd_base_before), float(pgd_base_after),
                    float(pgd_base_after - pgd_base_before),
                    int(applied_add), int(applied_del), float(deg_jsd)]

                for tgt in targets:
                    acc_t = eval_transfer_once(dgi, tgt, args, head_epochs=args.eval_head_epochs)
                    print(f"[transfer] epoch {epoch} | tgt={tgt} acc={acc_t:.4f}")
                    row.append(acc_t)

                csv_w.writerow(row); csv_f.flush()


            # DGI mode
            elif args.model_type == 0:
                dgi_optimizer.zero_grad()
                loss = dgi(features)
                loss.backward()
                dgi_optimizer.step()
                #loss = loss.item()
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), 'best_classification_pgd_{}.pkl'.format(args.model_type))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            #print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

        # create classifier model
        classifier = MultiClassifier(args.n_hidden, n_classes)
        if cuda:
            classifier.cuda()

        classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                                lr=args.classifier_lr,
                                                weight_decay=args.weight_decay)

        # flags used for transfer learning
        if args.data_src != args.data_id:
            pass
        else:
            dgi.load_state_dict(torch.load('best_classification_pgd_{}.pkl'.format(args.model_type)))

        with torch.no_grad():
            if args.model_type == 1:
                _, embeds, _ = dgi.forward(features)
            elif args.model_type == 2:
                embeds = dgi.encoder(features, corrupt=False)
            elif args.model_type == 0:
                embeds = dgi.encoder(features)
            else:
                dgi.eval()
                test_sampler = dgl.contrib.sampling.NeighborSampler(g, g.number_of_nodes(), -1,  # 0,
                                                                            neighbor_type='in', num_workers=1,
                                                                            add_self_loop=False,
                                                                            num_hops=args.n_layers + 1, shuffle=False)
                for nf in test_sampler:
                    nf.copy_from_parent()
                    embeds = dgi.encoder(nf, False)
                    print("test flow")

        embeds = embeds.detach()

        dur = []
        for epoch in range(args.n_classifier_epochs):
            classifier.train()
            if epoch >= 3:
                t0 = time.time()

            classifier_optimizer.zero_grad()
            preds = classifier(embeds)
            loss = F.nll_loss(preds[train_mask], labels[train_mask])
            # embed()
            loss.backward()
            classifier_optimizer.step()
            
            if epoch >= 3:
                dur.append(time.time() - t0)
            #acc = evaluate(classifier, embeds, labels, train_mask)
            #acc = evaluate(classifier, embeds, labels, val_mask)
            #print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
            #      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
            #                                    acc, n_edges / np.mean(dur) / 1000))

        # print()
        acc = evaluate(classifier, embeds, labels, test_mask)
        
        test_acc.append(acc)
        
    print("Test Accuracy {:.4f}, std {:.4f}".format(np.mean(test_acc), np.std(test_acc)))
    csv_f.close()
    print(f"=> Wrote log to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-2,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--model", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--model-type", type=int, default=2,
                    help="graph self-loop (default=False)")
    parser.add_argument("--graph-type", type=str, default="DD",
                    help="graph self-loop (default=False)")
    parser.add_argument("--data-id", type=str,default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--data-src", type=str, default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--file-path", type=str,
                        help="graph path")
    parser.add_argument("--label-path", type=str,
                        help="label path")
    parser.add_argument("--model-id", type=int, default=0,
                    help="[0, 1, 2, 3]")


    parser.add_argument("--pgd-steps", type=int, default=0,
                        help="PGD steps per epoch (0 disables adversarial training)")
    parser.add_argument("--pgd-add-k", type=int, default=10,
                        help="Number of edge additions per PGD step")
    parser.add_argument("--pgd-del-k", type=int, default=10,
                        help="Number of edge deletions per PGD step")
    parser.add_argument("--pgd-degree-cap", type=int, default=4,
                        help="Max degree change allowed per node (soft cap)")
    parser.add_argument("--pgd-rand-subset", type=int, default=200,
                        help="Random candidate subset size to score each step")

    parser.add_argument("--eval-targets", type=str, default="usa,brazil",
                    help="Comma-separated list of targets to eval each epoch")
    parser.add_argument("--eval-head-epochs", type=int, default=50,
                        help="Epochs for the linear head per eval")
    parser.add_argument("--logdir", type=str, default="logs_pgd",
                        help="Where to write CSV logs")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--features-recompute-per-epoch", type=int, default=0,
                        help="If 1, recompute degree features after crafting A_adv; else keep frozen")


    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    
    main(args)