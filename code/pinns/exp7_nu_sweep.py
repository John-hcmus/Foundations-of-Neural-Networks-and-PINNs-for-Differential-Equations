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


def run(nu, seed=0, adam_iters=20000, n_col=512, lbfgs_iters=2000, n_eval=20001):
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


def main():
    os.makedirs(OUT, exist_ok=True)
    nus = [1.0, 0.3, 0.1, 0.03, 0.01]
    rows = [run(nu) for nu in nus]
    print(f"{'nu':>7}{'be day lop':>12}{'||r||':>12}{'||e||':>12}"
          f"{'ti so':>12}{'chan 1/(pi^2 nu)':>18}{'thoa?':>8}{'hoi tu?':>9}{'eps_L2':>12}")
    for w in rows:
        ok = "co" if w["ty_so"] <= w["chan"] else "KHONG"
        hc = "co" if w["hoi_tu"] else "KHONG"
        print(f"{w['nu']:7g}{w['be_day_lop_bien']:12.4f}{w['r']:12.4e}{w['e']:12.4e}"
              f"{w['ty_so']:12.5f}{w['chan']:18.4f}{ok:>8}{hc:>9}"
              f"{w['eps_L2_tuong_doi']:12.4e}")

    print(f"\nChan duoc thoa o {sum(w['ty_so'] <= w['chan'] for w in rows)}/{len(rows)}"
          f" gia tri nu.")

    # Ti so co THUC SU tang theo 1/nu khong? Chi hoi quy tren cac diem ma huan
    # luyen da dua duoc phan du xuong du thap; o cac diem khac, ti so
    # ||e||/||r|| khong phai phep do cho hang so on dinh ma cho su that bai
    # cua chinh qua trinh toi uu.
    ht = [w for w in rows if w["hoi_tu"]]
    out = {"rows": rows}
    if len(ht) >= 2:
        A = np.log10([w["nu"] for w in ht]); B = np.log10([w["ty_so"] for w in ht])
        slope = float(np.polyfit(A, B, 1)[0])
        out["so_mu_hoi_tu"] = slope
        out["nu_hoi_tu"] = [w["nu"] for w in ht]
        ds = ", ".join("%g" % w["nu"] for w in ht)
        print(f"Hoi quy log-log tren {len(ht)} diem HOI TU (nu = {ds}):")
        print(f"  ti so ~ nu^({slope:.3f})        du bao cua chan: nu^(-1)")
    if len(ht) < len(rows):
        kh = [w for w in rows if not w["hoi_tu"]]
        print(f"Loai {len(kh)} diem khong hoi tu: "
              + ", ".join(f"nu={w['nu']:g} (||r||={w['r']:.2e})" for w in kh))
        A = np.log10([w["nu"] for w in rows]); B = np.log10([w["ty_so"] for w in rows])
        out["so_mu_tat_ca"] = float(np.polyfit(A, B, 1)[0])
        print(f"  (neu tinh ca chung: nu^({out['so_mu_tat_ca']:.3f}) -- vo nghia)")
    r0, r1 = rows[0], rows[-1]
    print(f"Sai so TUONG DOI tang {r1['eps_L2_tuong_doi']/r0['eps_L2_tuong_doi']:.0f} lan "
          f"khi nu giam {r0['nu']/r1['nu']:.0f} lan")
    json.dump(out, open(os.path.join(OUT, "exp7_nu_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
