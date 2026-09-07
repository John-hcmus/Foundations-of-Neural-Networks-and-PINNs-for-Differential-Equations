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
    Jr_cuoi = float((r.detach() ** 2).mean())
    return {"nu": nu, "r": nr, "e": ne, "ep": nex,
            "ty_so": ne / nr, "chan": 1.0 / (PI**2 * nu),
            "be_day_lop_bien": delta, "Jr_cuoi": Jr_cuoi,
            # nguong hoi tu: phan du phai duoc dua xuong du thap thi ti so
            # ||e||/||r|| moi la phep do co nghia cho hang so on dinh
            "hoi_tu": nr < 1e-2,
            "eps_L2_tuong_doi": ne / l2(ue, xs)}


SEEDS = [0, 1, 2]
NUS = [1.0, 0.3, 0.1, 0.03, 0.01]


def main():
    """Quet nu tren nhieu hat giong.

    Uoc luong so mu tu MOT hat giong khong on dinh (do duoc: -0,730 voi ngan
    sach 8000 vong va -0,535 voi 20000 vong, cung hat giong 0). Vi vay o day
    lap lai tren ba hat giong va bao cao TRUNG VI cung KHOANG, thay vi mot
    con so duy nhat.
    """
    import statistics as st
    os.makedirs(OUT, exist_ok=True)
    tat_ca, so_mu = {}, []
    for sd in SEEDS:
        rows = [run(nu, seed=sd) for nu in NUS]
        tat_ca[sd] = rows
        ht = [w for w in rows if w["hoi_tu"]]
        if len(ht) >= 2:
            A = np.log10([w["nu"] for w in ht]); B = np.log10([w["ty_so"] for w in ht])
            so_mu.append(float(np.polyfit(A, B, 1)[0]))

    print(f"{'nu':>7}{'be day lop':>12}{'ti so (trung vi)':>19}"
          f"{'khoang':>26}{'chan 1/(pi^2 nu)':>18}{'hoi tu':>9}")
    for j, nu in enumerate(NUS):
        ts = [tat_ca[sd][j]["ty_so"] for sd in SEEDS]
        hts = sum(tat_ca[sd][j]["hoi_tu"] for sd in SEEDS)
        w0 = tat_ca[SEEDS[0]][j]
        print(f"{nu:7g}{w0['be_day_lop_bien']:12.4f}{st.median(ts):19.5f}"
              f"{'[%.5f; %.5f]' % (min(ts), max(ts)):>26}"
              f"{1.0/(PI**2*nu):18.4f}{hts:>6}/{len(SEEDS)}")

    n_thoa = sum(w["ty_so"] <= w["chan"] for sd in SEEDS for w in tat_ca[sd])
    n_tong = len(SEEDS) * len(NUS)
    print(f"\nChan duoc thoa o {n_thoa}/{n_tong} truong hop (moi nu x moi hat giong).")
    print(f"So mu tren cac diem hoi tu, tung hat giong: "
          + ", ".join(f"{x:.3f}" for x in so_mu))
    print(f"  trung vi {st.median(so_mu):.3f}, khoang "
          f"[{min(so_mu):.3f}; {max(so_mu):.3f}]   (du bao cua chan: -1)")
    j = NUS.index(0.01)
    rr = [tat_ca[sd][j]["r"] for sd in SEEDS]
    print(f"Tai nu = 0,01: ||r|| = " + ", ".join(f"{x:.2e}" for x in rr)
          + "  -> khong hat giong nao hoi tu")
    json.dump({"chi_tiet": {str(k): v for k, v in tat_ca.items()},
               "so_mu_tung_hat_giong": so_mu,
               "so_mu_trung_vi": st.median(so_mu)},
              open(os.path.join(OUT, "exp7_nu_sweep.json"), "w"), indent=2)
if __name__ == "__main__":
    main()
