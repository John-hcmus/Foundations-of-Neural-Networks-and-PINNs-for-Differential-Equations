"""TN9 --- Chi phi tinh toan thuc do, va doi chieu voi phuong phap co dien.

Muc tieu MT4 o Chuong 1 hua bao cao "chi phi tinh toan thuc do". Huong dan
tai lieu tham khao cung yeu cau doi chieu sai so VA chi phi voi phuong phap
co dien, theo cach Raissi va cong su lam o Phan 4.1.

Thi nghiem do ba dai luong tren CUNG mot may, CUNG mot bai toan Burgers:

  (1) chi phi mot buoc huan luyen PINN, tach rieng ba thanh phan cua
      Nhan xet nx:chi-phi-bac-cao: luot xuoi thuan tuy, luot co dao ham bac
      hai, va luot co ca gradient theo tham so;
  (2) tong thoi gian huan luyen den khi dat sai so muc tieu;
  (3) thoi gian bo giai sai phan huu han dat CUNG muc sai so ay.

Diem (3) moi la doi chieu thuc su: no tra loi cau hoi "voi bai toan mot
chieu nay, PINN co re hon phuong phap co dien khong".
"""
from __future__ import annotations
import json, math, os, time

import numpy as np
import torch

from .core import DTYPE, FNN, grad, rel_l2, set_seed
from . import exp5_burgers as B

PI = math.pi
NU = 0.01 / PI
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _dem(f, n=20):
    """Thoi gian trung binh mot lan goi f, sau khi da lam nong."""
    for _ in range(3):
        f()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t0) / n


def chi_phi_mot_buoc(Nr=2500, seed=0):
    """Tach chi phi mot buoc thanh ba thanh phan."""
    set_seed(seed)
    net = FNN([2, 32, 32, 32, 32, 1], act="tanh")
    Xr, _, _ = B.sample(seed, Nr=Nr)
    params = list(net.parameters())

    def xuoi():
        with torch.no_grad():
            net(Xr)

    def phan_du():
        X = Xr.clone().requires_grad_(True)
        u = net(X)
        g1 = grad(u, X)
        uxx = grad(g1[:, 0:1], X)[:, 0:1]
        return ((g1[:, 1:2] + u * g1[:, 0:1] - NU * uxx) ** 2).mean()

    def buoc_day_du():
        J = phan_du()
        for p in params:
            p.grad = None
        J.backward()

    t_xuoi = _dem(xuoi)
    t_pd = _dem(phan_du)
    t_buoc = _dem(buoc_day_du)
    return {"luot_xuoi": t_xuoi, "phan_du": t_pd, "buoc_day_du": t_buoc,
            "boi_so_phan_du": t_pd / t_xuoi, "boi_so_buoc": t_buoc / t_xuoi,
            "so_tham_so": net.n_params(), "Nr": Nr}


def chi_phi_fd(muc_tieu, ref_min):
    """Thoi gian bo giai sai phan huu han dat sai so <= muc_tieu.

    Quet Nx tang dan; voi moi Nx do thoi gian va sai so so voi luoi min.
    """
    xm, tm, Um, _ = ref_min
    rows = []
    for Nx in (63, 127, 255, 511, 1023):
        t0 = time.perf_counter()
        x, ts, U, _ = B.reference(Nx=Nx)
        dt = time.perf_counter() - t0
        Ui = np.stack([np.interp(xm, x, U[i]) for i in range(len(ts))])
        e = float(np.linalg.norm(Ui - Um) / np.linalg.norm(Um))
        rows.append({"Nx": Nx, "giay": dt, "eps_L2": e, "dat": e <= muc_tieu})
        if e <= muc_tieu:
            break
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 1. Chi phi mot buoc huan luyen PINN (Burgers, Nr = 2500) ===")
    c = chi_phi_mot_buoc()
    print(f"  luot xuoi thuan tuy          {c['luot_xuoi']*1e3:8.2f} ms")
    print(f"  + dao ham bac hai (phan du)  {c['phan_du']*1e3:8.2f} ms"
          f"   = {c['boi_so_phan_du']:5.1f}x luot xuoi")
    print(f"  + gradient theo tham so      {c['buoc_day_du']*1e3:8.2f} ms"
          f"   = {c['boi_so_buoc']:5.1f}x luot xuoi")
    print(f"  (Nhan xet nx:chi-phi-bac-cao du bao boi so 4-6 lan)")

    print("\n=== 2. Tong thoi gian huan luyen PINN (TN5, trong so co dinh) ===")
    ref_tmp = B.reference(Nx=511)      # luoi tho, chi de ham run() co cho danh gia
    t0 = time.perf_counter()
    B.run(None, seed=0, ref=ref_tmp)
    t_pinn = time.perf_counter() - t0
    print(f"  6000 vong Adam + 800 vong L-BFGS: {t_pinn:.1f} s")

    print("\n=== 3. Nghiem tham chieu min (Nx = 2047) lam chuan ===")
    t0 = time.perf_counter()
    ref_min = B.reference(Nx=2047)
    t_min = time.perf_counter() - t0
    print(f"  Nx = 2047: {t_min:.1f} s")

    print("\n=== 4. Chi phi PINN so voi sai phan huu han, cung muc sai so ===")
    # sai so PINN da do o TN5 (trong so co dinh, 800 vong L-BFGS)
    f = os.path.join(OUT, "exp5_burgers.json")
    eps_pinn = None
    if os.path.exists(f):
        d = json.load(open(f))
        eps_pinn = d["ket_qua"][0]["eps_L2_cuoi"]
    if eps_pinn is None:
        eps_pinn = 8.19e-2
    print(f"  sai so PINN dat duoc: {eps_pinn:.4e}")
    rows = chi_phi_fd(eps_pinn, ref_min)
    print(f"  {'Nx':>6}{'giay':>10}{'eps_L2':>12}   dat muc PINN?")
    for r in rows:
        print(f"  {r['Nx']:6d}{r['giay']:10.3f}{r['eps_L2']:12.4e}"
              f"   {'co' if r['dat'] else 'khong'}")
    dat = [r for r in rows if r["dat"]]
    out = {"mot_buoc": c, "fd": rows, "eps_pinn": eps_pinn,
           "giay_ref_min": t_min, "giay_huan_luyen_pinn": t_pinn}
    if dat:
        r = dat[0]
        print(f"\n  Sai phan huu han dat cung muc sai so voi Nx = {r['Nx']} "
              f"trong {r['giay']:.3f} s.")
        print(f"  PINN can {t_pinn:.1f} s  ->  cham hon {t_pinn/r['giay']:.0f} lan.")
        print(f"  LUU Y: bo giai FD dung dt = 5e-5 co dinh (chon cho Nx = 2047),")
        print(f"  qua nho so voi nhu cau cua Nx = {r['Nx']}. Chi phi FD that su con")
        print(f"  thap hon, nen ti so tren la uoc luong CO LOI cho PINN.")
        out["fd_dat"] = r
        out["cham_hon_lan"] = t_pinn / r["giay"]
    json.dump(out, open(os.path.join(OUT, "exp9_chiphi.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
