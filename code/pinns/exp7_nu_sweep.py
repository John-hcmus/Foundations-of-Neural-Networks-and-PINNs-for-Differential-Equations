"""TN7 --- Su suy bien cua hang so on dinh theo 1/nu (Menh de md:chan-suy-bien).

Bai toan doi luu--khuech tan, chinh la bai toan cua menh de:

    -nu u'' + u' = 1 tren (0,1),  u(0) = u(1) = 0.

Nghiem dung co dang dong, voi mot LOP BIEN be day O(nu) gan x = 1:

    u*(x) = x - (e^{(x-1)/nu} - e^{-1/nu}) / (1 - e^{-1/nu})

(viet o dang nay de on dinh so khi nu nho; dang thong thuong
(1 - e^{x/nu})/(e^{1/nu} - 1) bi tran so voi nu <= 0,05).

Menh de khang dinh, voi dieu kien bien thoa CHINH XAC:
    ||e||_L2 <= 1/(pi^2 nu) ||r||_L2

Hai cau hoi:
  (1) chan co dung tren toan dai nu khong?
  (2) ti so ||e||/||r|| co THUC SU tang theo 1/nu khong, hay chan chi long ra
      con thuc te khong doi? Cau hoi thu hai moi la noi dung thuc chat, vi no
      quyet dinh xem tang thu nhat cua Bang tab:ba-nguyen-nhan co that hay khong.

Dung rang buoc cung u = x(1-x) N(x) de gia thiet cua menh de duoc thoa theo
cau truc, dung nhu TN6.
"""
from __future__ import annotations
import json, math, os
import numpy as np
import torch
from .core import DTYPE, FNN, grad, set_seed, train_adam, train_lbfgs, uniform_1d

PI = math.pi
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def u_exact(x, nu):
    em = math.exp(-1.0 / nu)
    return x - (torch.exp((x - 1.0) / nu) - em) / (1.0 - em)


def du_exact(x, nu):
    em = math.exp(-1.0 / nu)
    return 1.0 - torch.exp((x - 1.0) / nu) / (nu * (1.0 - em))


def l2(vals, x):
    return torch.sqrt(torch.trapz(vals.squeeze() ** 2, x.squeeze())).item()


def run(nu, seed=0, adam_iters=8000, n_col=512, lbfgs_iters=800, n_eval=20001):
    set_seed(seed)
    net = FNN([1, 64, 64, 64, 1], act="tanh")
    model = lambda x: x * (1 - x) * net(x)          # Bo de bd:rang-buoc-cung
    xr = uniform_1d(n_col)
    xe = uniform_1d(n_eval)

    def loss_fn():
        x = xr.clone().requires_grad_(True)
        u = model(x)
        ux = grad(u, x, 1)
        uxx = grad(ux, x, 1)
        Jr = ((-nu * uxx + ux - 1.0) ** 2).mean()
        return Jr, {"Jr": Jr.item()}

    train_adam(net, loss_fn, adam_iters)
    train_lbfgs(net, loss_fn, max_iter=lbfgs_iters)

    x = xe.clone().requires_grad_(True)
    u = model(x)
    ux = grad(u, x, 1)
    uxx = grad(ux, x, 1)
    r = -nu * uxx + ux - 1.0
    e = u - u_exact(x, nu)
    ex = ux - du_exact(x, nu)
    xs = xe.detach()
    nr, ne, nex = l2(r.detach(), xs), l2(e.detach(), xs), l2(ex.detach(), xs)
    # be day lop bien do duoc: 2 max|u| / max|u_x| tren nghiem DUNG
    with torch.no_grad():
        ue, due = u_exact(xs, nu), du_exact(xs, nu)
        delta = 2 * ue.abs().max().item() / due.abs().max().item()
    return {"nu": nu, "r": nr, "e": ne, "ep": nex,
            "ty_so": ne / nr, "chan": 1.0 / (PI**2 * nu),
            "be_day_lop_bien": delta,
            "eps_L2_tuong_doi": ne / l2(ue, xs)}


def main():
    os.makedirs(OUT, exist_ok=True)
    nus = [1.0, 0.3, 0.1, 0.03, 0.01]
    rows = [run(nu) for nu in nus]
    print(f"{'nu':>7}{'be day lop':>12}{'||r||':>12}{'||e||':>12}"
          f"{'ti so':>12}{'chan 1/(pi^2 nu)':>18}{'thoa?':>8}{'eps_L2':>12}")
    for w in rows:
        ok = "co" if w["ty_so"] <= w["chan"] else "KHONG"
        print(f"{w['nu']:7g}{w['be_day_lop_bien']:12.4f}{w['r']:12.4e}{w['e']:12.4e}"
              f"{w['ty_so']:12.5f}{w['chan']:18.4f}{ok:>8}{w['eps_L2_tuong_doi']:12.4e}")
    # ti so co tang theo 1/nu khong? hoi quy log-log
    A = np.log10([w["nu"] for w in rows])
    B = np.log10([w["ty_so"] for w in rows])
    slope = float(np.polyfit(A, B, 1)[0])
    print(f"\nhoi quy log-log: ti so ~ nu^({slope:.3f})   (du bao cua chan: nu^(-1))")
    r0, r1 = rows[0], rows[-1]
    print(f"ti so tang {r1['ty_so']/r0['ty_so']:.1f} lan khi nu giam "
          f"{r0['nu']/r1['nu']:.0f} lan")
    print(f"sai so tuong doi tang {r1['eps_L2_tuong_doi']/r0['eps_L2_tuong_doi']:.1f} lan")
    json.dump({"rows": rows, "so_mu": slope},
              open(os.path.join(OUT, "exp7_nu_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
