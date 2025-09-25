# linearPredictor.py  (logging-enabled)
import os, csv, time, argparse, pickle
import numpy as np, networkx as nx, torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args
from models.dgi import DGI, MultiClassifier
from models.subgi import SubGI
import scipy.sparse as sp
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding

# ==================== helpers: logging ====================

class ResultsLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.per_seed_path = os.path.join(self.out_dir, "per_seed.csv")
        self.summary_path  = os.path.join(self.out_dir, "summary.csv")
        self._ensure_headers()

    def _ensure_headers(self):
        per_seed_cols = [
            # identifiers
            "timestamp","run_tag","run_idx","data_id","data_src","file_path","label_path",
            # config
            "model_type","model_id","feature_mode","bypass_encoder","residualize_flag",
            "probe_degree_flag","n_hidden","n_layers","n_dgi_epochs","n_classifier_epochs",
            # quick feature stats
            "feat_mean","feat_std",
            # metrics
            "spearman_deg_label","probe_R2_logdeg","probe_acc_deg_quartile",
            "acc_Z","acc_Z_perp","acc_drop","baseline_deg_bin"
        ]
        summary_cols = [
            "timestamp","run_tag","data_id","data_src","file_path","label_path",
            "feature_mode","bypass_encoder","residualize_flag","probe_degree_flag",
            "n_hidden","n_layers","n_dgi_epochs","n_classifier_epochs",
            "mean_acc_test","std_acc_test","num_runs"
        ]
        if not os.path.exists(self.per_seed_path):
            with open(self.per_seed_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=per_seed_cols).writeheader()
        if not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=summary_cols).writeheader()
        self.per_seed_cols = per_seed_cols
        self.summary_cols  = summary_cols

    def log_per_seed(self, row: dict):
        row = {k: row.get(k, None) for k in self.per_seed_cols}
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.per_seed_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.per_seed_cols).writerow(row)

    def log_summary(self, row: dict):
        row = {k: row.get(k, None) for k in self.summary_cols}
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.summary_cols).writerow(row)

# ==================== math helpers ====================

def _zscore(x): return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True).clamp_min(1e-8))

def get_deg_pr(g_dgl):
    deg = g_dgl.in_degrees().float()
    src, dst = g_dgl.edges()
    H = nx.DiGraph()
    H.add_nodes_from(range(g_dgl.number_of_nodes()))
    H.add_edges_from(zip(src.tolist(), dst.tolist()))
    pr_dict = nx.pagerank(H, alpha=0.85, tol=1e-8, max_iter=1000)
    pr = torch.tensor([pr_dict[i] for i in range(H.number_of_nodes())], dtype=torch.float32)
    return deg, pr

def to_quartiles(x: torch.Tensor, k=4):
    qs = torch.quantile(x, torch.linspace(0,1,k+1))
    qs = torch.unique_consecutive(qs)
    return torch.bucketize(x, qs[1:-1])

def degree_probe_metrics(Z: torch.Tensor, g_dgl):
    Z = Z.detach()
    deg, _ = get_deg_pr(g_dgl)
    y = torch.log1p(deg).view(-1,1)
    X = torch.cat([Z, torch.ones(Z.size(0),1)], dim=1)
    lam = 1e-3
    W = torch.linalg.solve(X.T @ X + lam*torch.eye(X.size(1)), X.T @ y)
    yhat = X @ W
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    R2 = (1 - ss_res/ss_tot).item()
    deg_q = to_quartiles(deg, 4)
    clf = nn.Linear(Z.size(1), 4)
    opt = torch.optim.LBFGS(clf.parameters(), lr=0.5, max_iter=200)
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = ce(clf(Z), deg_q); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        acc_q = (clf(Z).argmax(1) == deg_q).float().mean().item()
    return {"R2_logdeg": R2, "Acc_deg_quartile": acc_q}

def residualize_Z(Z: torch.Tensor, g_dgl, include_pr=True):
    deg, pr = get_deg_pr(g_dgl)
    cols = [torch.ones_like(deg), torch.log1p(deg), deg, deg**0.5, deg**2]
    if include_pr: cols.append(pr)
    F = torch.stack(cols, dim=1)
    Fz = _zscore(F)
    P = Fz @ torch.linalg.pinv(Fz.T @ Fz) @ Fz.T
    return Z - P @ Z

def few_shot_mask(labels: torch.Tensor, test_mask: torch.Tensor, shots_per_class: int):
    y = labels.cpu().numpy()
    idx = np.arange(len(y))
    keep = []
    for c in np.unique(y):
        pool = idx[(y==c) & (~test_mask.cpu().numpy())]
        if len(pool)==0: continue
        take = min(shots_per_class, len(pool))
        keep += np.random.choice(pool, size=take, replace=False).tolist()
    m = torch.zeros_like(test_mask, dtype=torch.bool)
    m[keep] = True
    return m

def clustering_quartile_labels_from_dgl(g_dgl):
    s, t = g_dgl.edges()
    G = nx.Graph()
    G.add_nodes_from(range(g_dgl.number_of_nodes()))
    G.add_edges_from(zip(s.tolist(), t.tolist()))
    cc = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float32)
    return to_quartiles(cc, 4)

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
        A[id_a, id_b] = 1
    embedding = SpectralEmbedding(n_components=args.n_hidden)
    features = torch.FloatTensor(embedding.fit_transform(A))
    return features

def degree_bucketing(graph, args, degree_emb=None, max_degree = 10):
    max_degree = args.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

def createTraining(labels, valid_mask = None, train_ratio=0.8):
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask

def read_struct_net(args):
    g = nx.Graph()
    with open(args.file_path) as IN:
        for line in IN:
            u,v = line.strip().split()
            g.add_edge(int(u), int(v))
    labels = {}
    with open(args.label_path) as IN:
        IN.readline()
        for line in IN:
            a,b = line.strip().split(' ')
            labels[int(a)] = int(b)
    return g, labels

def constructDGL(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
    return new_g, relabels

def output_adj(graph):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1
    return A

# ==================== main ====================

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logger = ResultsLogger(args.out_dir)

    torch.manual_seed(2)  # keep as in your original script
    test_acc = []

    for runs in tqdm(range(100)):
        g_nx, labels_dict = read_struct_net(args)
        g_nx.remove_edges_from(nx.selfloop_edges(g_nx))
        g, labels_list = constructDGL(g_nx, labels_dict)

        if args.label_mode == "clustering":
            labels_list = clustering_quartile_labels_from_dgl(g).tolist()

        labels = torch.LongTensor(labels_list)

        degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0,1,[100, args.n_hidden])),
                                  requires_grad=False)

        if args.feature_mode == "degree":
            features = degree_bucketing(g, args, degree_emb)
        elif args.feature_mode == "permute_degree":
            f = degree_bucketing(g, args, degree_emb)
            perm = torch.randperm(f.size(0))
            features = f[perm]
        elif args.feature_mode == "constant":
            features = torch.ones(g.number_of_nodes(), args.n_hidden)
        elif args.feature_mode == "spectral":
            features = spectral_feature(g, args)
        else:
            raise ValueError(f"unknown feature_mode={args.feature_mode}")

        feat_mean = float(features.mean())
        feat_std  = float(features.std())
        print(f"[{args.data_id or 'target'}] feature_mode={args.feature_mode} mean={feat_mean:.4f} std={feat_std:.4f}")

        train_mask, test_mask = createTraining(labels, None)
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(train_mask)
            test_mask = torch.BoolTensor(test_mask)
        else:
            train_mask = torch.ByteTensor(train_mask)
            test_mask = torch.ByteTensor(test_mask)

        in_feats = features.shape[1]
        n_classes = labels.max().item() + 1

        cuda = False
        g.readonly()

        # build encoder
        if args.model_type == 1:
            dgi = VGAE(g, in_feats, args.n_hidden, args.n_hidden, args.dropout)
            dgi.prepare()
            dgi.adj_train = sp.csr_matrix(output_adj(g))
        elif args.model_type == 0:
            dgi = DGI(g, in_feats, args.n_hidden, args.n_layers, nn.PReLU(args.n_hidden), args.dropout)
        elif args.model_type == 2:
            dgi = SubGI(g, in_feats, args.n_hidden, args.n_layers, nn.PReLU(args.n_hidden), args.dropout, args.model_id)
        else:
            raise ValueError("unknown model_type")
        if cuda: dgi.cuda()

        dgi_optimizer = torch.optim.Adam(dgi.parameters(), lr=args.dgi_lr, weight_decay=args.weight_decay)

        # train encoder if requested
        g.ndata['features'] = features
        best = float("inf"); cnt_wait = 0
        for epoch in range(args.n_dgi_epochs):
            train_sampler = dgl.contrib.sampling.NeighborSampler(
                g, 256, 5, neighbor_type='in', num_workers=1, add_self_loop=False,
                num_hops=args.n_layers + 1, shuffle=True)
            dgi.train()
            loss = 0.0
            if args.model_type == 1:
                dgi.optimizer = dgi_optimizer; dgi.train_sampler = train_sampler; dgi.features = features
                loss = dgi.train_model()
            elif args.model_type == 2:
                for nf in train_sampler:
                    dgi_optimizer.zero_grad()
                    l = dgi(features, nf)
                    l.backward(); loss += l
                    dgi_optimizer.step()
            elif args.model_type == 0:
                dgi_optimizer.zero_grad()
                loss = dgi(features)
                loss.backward()
                dgi_optimizer.step()
            if loss < best:
                best = loss; cnt_wait = 0
                torch.save(dgi.state_dict(), 'best_classification_linear_{}.pkl'.format(args.model_type))
            else:
                cnt_wait += 1
            if cnt_wait == args.patience:
                print('Early stopping!')
                break

        # classifier
        classifier = MultiClassifier(args.n_hidden, n_classes)
        if cuda: classifier.cuda()
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay)

        # load best encoder (even if n_dgi_epochs==0 it’s harmless)
        dgi.load_state_dict(torch.load('best_classification_linear_{}.pkl'.format(args.model_type)))

        with torch.no_grad():
            if args.bypass_encoder:
                embeds = features.clone()
            else:
                if args.model_type == 1:
                    _, embeds, _ = dgi.forward(features)
                elif args.model_type == 2:
                    embeds = dgi.encoder(features, corrupt=False)
                elif args.model_type == 0:
                    embeds = dgi.encoder(features)
        embeds = embeds.detach()

        # metrics: Spearman
        from scipy.stats import spearmanr
        deg = g.in_degrees().numpy()
        rho, _ = spearmanr(deg[test_mask.numpy()], labels[test_mask].numpy())
        print("Spearman(label, degree) on TEST =", rho)

        # optional probe
        probe_R2 = None; probe_accq = None
        if args.probe_degree and not args.bypass_encoder:
            probe = degree_probe_metrics(embeds, g)
            probe_R2 = float(probe['R2_logdeg']); probe_accq = float(probe['Acc_deg_quartile'])
            print(f"[probe] R2_logdeg={probe_R2:.3f}  Acc_degQ={probe_accq:.3f}")

        # choose train subset
        train_mask_use = train_mask

        # train classifier on Z
        for _ in range(args.n_classifier_epochs):
            classifier.train()
            classifier_optimizer.zero_grad()
            preds = classifier(embeds)
            loss = F.nll_loss(preds[train_mask_use], labels[train_mask_use])
            loss.backward()
            classifier_optimizer.step()

        acc_Z = evaluate(classifier, embeds, labels, test_mask)
        print(f"[eval] Acc(Z) = {acc_Z:.4f}")

        # residualized eval
        acc_perp = None; acc_drop = None
        if args.residualize and not args.bypass_encoder:
            Z_perp = residualize_Z(embeds, g, include_pr=True)
            classifier_perp = MultiClassifier(args.n_hidden, n_classes)
            if cuda: classifier_perp.cuda()
            optp = torch.optim.Adam(classifier_perp.parameters(), lr=args.classifier_lr, weight_decay=args.weight_decay)
            for _ in range(args.n_classifier_epochs):
                classifier_perp.train(); optp.zero_grad()
                preds = classifier_perp(Z_perp)
                loss = F.nll_loss(preds[train_mask_use], labels[train_mask_use])
                loss.backward(); optp.step()
            acc_perp = evaluate(classifier_perp, Z_perp, labels, test_mask)
            acc_drop = acc_Z - acc_perp
            print(f"[eval] Acc(Z_perp | degree/PR residualized) = {acc_perp:.4f}  (drop {acc_drop:+.4f})")

        # majority-per-degree-bin baseline
        train_bins = features[train_mask].argmax(1).cpu().numpy()
        test_bins  = features[test_mask].argmax(1).cpu().numpy()
        y_train    = labels[train_mask].cpu().numpy()
        y_test     = labels[test_mask].cpu().numpy()
        groups = defaultdict(list)
        for b, y in zip(train_bins, y_train):
            groups[b].append(y)
        bin2lab = {b: Counter(ys).most_common(1)[0][0] for b, ys in groups.items()}
        global_majority = Counter(y_train).most_common(1)[0][0]
        y_pred = np.array([bin2lab.get(b, global_majority) for b in test_bins])
        baseline_acc = float((y_pred == y_test).mean())
        print("Majority-per-degree-bin baseline acc:", baseline_acc)

        # log per-seed
        logger.log_per_seed({
            "run_tag": args.run_tag,
            "run_idx": runs,
            "data_id": args.data_id, "data_src": args.data_src,
            "file_path": args.file_path, "label_path": args.label_path,
            "model_type": args.model_type, "model_id": args.model_id,
            "feature_mode": args.feature_mode,
            "bypass_encoder": bool(args.bypass_encoder),
            "residualize_flag": bool(args.residualize),
            "probe_degree_flag": bool(args.probe_degree),
            "n_hidden": args.n_hidden, "n_layers": args.n_layers,
            "n_dgi_epochs": args.n_dgi_epochs, "n_classifier_epochs": args.n_classifier_epochs,
            "feat_mean": feat_mean, "feat_std": feat_std,
            "spearman_deg_label": float(rho),
            "probe_R2_logdeg": probe_R2,
            "probe_acc_deg_quartile": probe_accq,
            "acc_Z": float(acc_Z),
            "acc_Z_perp": (None if acc_perp is None else float(acc_perp)),
            "acc_drop": (None if acc_drop is None else float(acc_drop)),
            "baseline_deg_bin": baseline_acc,
        })

        test_acc.append(acc_Z)

    # summary row
    mean_acc = float(np.mean(test_acc))
    std_acc  = float(np.std(test_acc))
    print("Test Accuracy {:.4f}, std {:.4f}".format(mean_acc, std_acc))

    logger.log_summary({
        "run_tag": args.run_tag,
        "data_id": args.data_id, "data_src": args.data_src,
        "file_path": args.file_path, "label_path": args.label_path,
        "feature_mode": args.feature_mode,
        "bypass_encoder": bool(args.bypass_encoder),
        "residualize_flag": bool(args.residualize),
        "probe_degree_flag": bool(args.probe_degree),
        "n_hidden": args.n_hidden, "n_layers": args.n_layers,
        "n_dgi_epochs": args.n_dgi_epochs, "n_classifier_epochs": args.n_classifier_epochs,
        "mean_acc_test": mean_acc, "std_acc_test": std_acc, "num_runs": len(test_acc),
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dgi-lr", type=float, default=1e-2)
    parser.add_argument("--classifier-lr", type=float, default=1e-2)
    parser.add_argument("--n-dgi-epochs", type=int, default=300)
    parser.add_argument("--n-classifier-epochs", type=int, default=100)
    parser.add_argument("--n-hidden", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--model", action='store_true')
    parser.add_argument("--self-loop", action='store_true')
    parser.add_argument("--model-type", type=int, default=2)
    parser.add_argument("--graph-type", type=str, default="DD")
    parser.add_argument("--data-id", type=str, default='')
    parser.add_argument("--data-src", type=str, default='')
    parser.add_argument("--file-path", type=str, required=True)
    parser.add_argument("--label-path", type=str, required=True)
    parser.add_argument("--model-id", type=int, default=0)
    # === PROBE FLAGS ===
    parser.add_argument("--feature-mode", type=str, default="degree",
        choices=["degree","constant","permute_degree","spectral"])
    parser.add_argument("--bypass-encoder", action="store_true")
    parser.add_argument("--probe-degree", action="store_true")
    parser.add_argument("--residualize", action="store_true")
    parser.add_argument("--few-shot-shots", type=int, default=0)
    parser.add_argument("--label-mode", type=str, default="original",
        choices=["original","clustering"])
    # === NEW: logging flags ===
    parser.add_argument("--out-dir", type=str, default="results",
        help="Directory to write per_seed.csv and summary.csv")
    parser.add_argument("--run-tag", type=str, default="",
        help="Free-form tag to label this invocation (e.g., normal/residualized/permute/bypass)")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    main(args)
