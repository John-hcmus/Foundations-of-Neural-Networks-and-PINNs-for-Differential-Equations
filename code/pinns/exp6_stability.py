"""TN6 --- Kiem chung chan on dinh cua Menh de md:chan-on-dinh.

Bai toan Poisson mot chieu (giong TN1):

    -u'' = f tren (0,1),   u(0) = u(1) = 0,   u*(x) = sin(2 pi x),
    f(x) = 4 pi^2 sin(2 pi x).

Menh de khang dinh, VOI GIA THIET dieu kien bien duoc thoa CHINH XAC:

    ||e'||_L2 <= (1/pi)   ||r||_L2
    ||e ||_L2 <= (1/pi^2) ||r||_L2

Thi nghiem do dong thoi hai chuan tren mot luoi danh gia min, tai nhieu muc
do chinh xac khac nhau (cac moc trong qua trinh huan luyen), roi kiem tra ty
so ||e||/||r|| co nam duoi hang so 1/pi^2 hay khong.

Chay ba cau hinh de tach vai tro cua gia thiet:
  (a) rang buoc CUNG   u = x(1-x) N(x)       -> gia thiet duoc thoa CHINH XAC
  (b) rang buoc MEM, lam_b = 100             -> gia thiet thoa gan dung
  (c) rang buoc MEM, lam_b = 0.01            -> gia thiet bi vi pham nang
"""
from __future__ import annotations

import json
import math
import os

import torch

from .core import DTYPE, FNN, grad, set_seed, train_adam, train_lbfgs, uniform_1d

PI = math.pi
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def u_exact(x):
    return torch.sin(2 * PI * x)


def du_exact(x):
    return 2 * PI * torch.cos(2 * PI * x)


def f_rhs(x):
    return 4 * PI**2 * torch.sin(2 * PI * x)


def make_model(net, hard: bool):
    """Tra ve ham nghiem thu. hard=True dung Bo de bd:rang-buoc-cung."""
    if hard:
        return lambda x: x * (1 - x) * net(x)
    return lambda x: net(x)


def l2_norm(vals, x):
    """Chuan L2(0,1) bang quy tac hinh thang tren luoi deu x."""
    return torch.sqrt(torch.trapz(vals.squeeze() ** 2, x.squeeze())).item()


def measure(model, xe):
    """Do ||r||, ||e||, ||e'|| tren luoi danh gia xe."""
    xe = xe.clone().requires_grad_(True)
    u = model(xe)
    u_x = grad(u, xe, 1)
    u_xx = grad(u_x, xe, 1)
    r = -u_xx - f_rhs(xe)
    e = u - u_exact(xe)
    e_x = u_x - du_exact(xe)
    xs = xe.detach()
    with torch.no_grad():
        xb = torch.tensor([[0.0], [1.0]], dtype=xs.dtype)
        eb = model(xb).abs().max().item()
    return (l2_norm(r.detach(), xs), l2_norm(e.detach(), xs),
            l2_norm(e_x.detach(), xs), eb)


def run(hard: bool, seed: int = 0, adam_iters: int = 4000, n_col: int = 256,
        lam_b: float = 100.0, snapshots=(0, 50, 200, 1000, 4000)):
    set_seed(seed)
    net = FNN([1, 32, 32, 32, 1], act="tanh")
    model = make_model(net, hard)

    xr = uniform_1d(n_col)          # quy uoc luan van: ke ca diem bien
    xb = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    xe = torch.linspace(0, 1, 20001, dtype=DTYPE).reshape(-1, 1)

    def loss_fn():
        x = xr.clone().requires_grad_(True)
        u = model(x)
        u_xx = grad(u, x, 2)
        Jr = ((-u_xx - f_rhs(x)) ** 2).mean()
        if hard:
            return Jr, {"Jr": Jr.item(), "Jb": 0.0}
        Jb = (model(xb) ** 2).mean()
        return Jr + lam_b * Jb, {"Jr": Jr.item(), "Jb": Jb.item()}

    rows = []

    def snap(tag):
        nr, ne, nex, eb = measure(model, xe)
        rows.append({
            "moc": tag, "r": nr, "e": ne, "ep": nex, "sai_so_bien": eb,
            "ty_so_e": ne / nr if nr > 0 else float("nan"),
            "ty_so_ep": nex / nr if nr > 0 else float("nan"),
        })

    snap("khoi tao")
    prev = 0
    for k in snapshots:
        if k == 0:
            continue
        train_adam(net, loss_fn, k - prev)
        prev = k
        snap(f"Adam {k}")
    train_lbfgs(net, loss_fn, max_iter=500)
    snap("+ L-BFGS")
    return rows


CAU_HINH = [
    ("rang_buoc_cung",     dict(hard=True)),
    ("rang_buoc_mem_100",  dict(hard=False, lam_b=100.0)),
    ("rang_buoc_mem_0.01", dict(hard=False, lam_b=0.01)),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for key, kw in CAU_HINH:
        rows = run(**kw)
        out[key] = rows
        print(f"\n=== {key} ===")
        print(f"{'moc':<12}{'||r||':>12}{'||e||':>12}{'sai so bien':>13}"
              f"{'||e||/||r||':>14}{'1/pi^2':>10}  thoa?")
        for w in rows:
            ok = "co" if w["ty_so_e"] <= 1 / PI**2 else "KHONG"
            print(f"{w['moc']:<12}{w['r']:12.4e}{w['e']:12.4e}"
                  f"{w['sai_so_bien']:13.4e}{w['ty_so_e']:14.6f}"
                  f"{1/PI**2:10.6f}  {ok}")
    with open(os.path.join(OUT, "exp6_stability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nDa ghi {os.path.join(OUT, 'exp6_stability.json')}")


if __name__ == "__main__":
    main()
