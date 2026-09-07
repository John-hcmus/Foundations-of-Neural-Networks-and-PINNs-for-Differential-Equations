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
import json, math, os, statistics as st, time

import numpy as np
import torch

from .core import DTYPE, FNN, grad, rel_l2, set_seed
from . import exp5_burgers as B

PI = math.pi
NU = 0.01 / PI
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _dem(f, n=20, vong=9):
    """Thoi gian mot lan goi f: TRUNG VI cua `vong` dot do, moi dot `n` lan.

    Do trung binh cua mot dot duy nhat qua nhieu nhieu: giua cac lan chay
    doc lap, ti so "phan du / luot xuoi" dao dong tu 4,1 den 5,1 chi vi
    luot xuoi (co 2 ms) rat nhay voi tai he thong. Trung vi cua chin dot
    on dinh hon han.
    """
    for _ in range(5):
        f()
    dot = []
    for _ in range(vong):
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        dot.append((time.perf_counter() - t0) / n)
    return st.median(dot)


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
            "c_menh_de": t_buoc / t_pd,
            "so_tham_so": net.n_params(), "Nr": Nr}


def quet_kich_thuoc(seed=0):
    """Do chi phi mot buoc o nhieu kich thuoc lo.

    Ly do phai quet: voi Nr = 2500 luot xuoi chi ton ~1,3 ms, qua ngan de do
    tin cay tren mot may dung chung -- ba lan chay doc lap cho boi so 4,2 /
    7,2 / 7,4. Tang Nr len lam luot xuoi du dai de phep do on dinh, va ti so
    thi gan nhu khong doi theo Nr. Ta lay Nr lon nhat lam so bao cao.
    """
    return [chi_phi_mot_buoc(Nr=n, seed=seed) for n in (2500, 10000, 40000)]


def _do_fd(Nx, dt, ref_min):
    """Chay bo giai FD mot lan, tra ve (giay, eps_L2) hoac None neu luoi
    thoi gian khong khop voi luoi cua nghiem tham chieu."""
    xm, tm, Um, _ = ref_min
    t0 = time.perf_counter()
    x, ts, U, _ = B.reference(Nx=Nx, dt=dt)
    el = time.perf_counter() - t0
    if len(ts) != len(tm):
        return None
    Ui = np.stack([np.interp(xm, x, U[i]) for i in range(len(ts))])
    e = float(np.linalg.norm(Ui - Um) / np.linalg.norm(Um))
    return el, e


def chi_phi_fd(muc_tieu, ref_min):
    """Thoi gian bo giai sai phan huu han dat sai so <= muc_tieu.

    Quet Nx tang dan, GIU dt = 5e-5 nhu bo giai tham chieu. Day la phep do
    "ngay tho": dt ay duoc chon cho Nx = 2047 nen qua nho voi luoi tho.
    """
    rows = []
    for Nx in (15, 31, 63, 127, 255, 511, 1023):
        r = _do_fd(Nx, 5e-5, ref_min)
        if r is None:
            continue
        el, e = r
        rows.append({"Nx": Nx, "dt": 5e-5, "giay": el, "eps_L2": e,
                     "dat": e <= muc_tieu})
        if e <= muc_tieu:
            break
    return rows


def chi_phi_fd_dt_hop_ly(Nx, ref_min):
    """Chi phi FD that su tai luoi Nx, khi dt duoc chon cho CHINH luoi ay.

    Bo giai la RK4 tuong minh, nen dt bi chan boi hai dieu kien on dinh:
        khuech tan  dt <= dx^2 / (2 nu),
        doi luu     dt <= dx / max|u|  (max|u| <= 1 voi bai toan nay).
    Ta quet dt tang dan trong vung on dinh va bao cao lan chay nhanh nhat
    van dat cung muc sai so. Sai so khong doi khi dt giam, chung to sai so
    KHONG GIAN moi la thanh phan chiem uu the -- day chinh la ly do dt = 5e-5
    la lang phi tren luoi tho.
    """
    dx = 2.0 / (Nx + 1)
    chan_kt = dx * dx / (2.0 * NU)
    chan_dl = dx / 1.0
    chan = min(chan_kt, chan_dl)
    rows = []
    for dt in (5e-5, 1e-3, 5e-3, 1e-2):
        if dt > chan / 2:          # giu bien an toan gap doi
            continue
        r = _do_fd(Nx, dt, ref_min)
        if r is None:
            continue
        el, e = r
        rows.append({"Nx": Nx, "dt": dt, "n_buoc": int(round(1.0 / dt)),
                     "giay": el, "eps_L2": e})
    return {"chan_khuech_tan": chan_kt, "chan_doi_luu": chan_dl,
            "lan_chay": rows}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 1. Chi phi mot buoc huan luyen PINN (Burgers) ===")
    quet = quet_kich_thuoc()
    print(f"  {'Nr':>7}{'xuoi (ms)':>12}{'phan du (ms)':>15}{'buoc (ms)':>12}"
          f"{'pd/xuoi':>10}{'c=buoc/pd':>11}")
    for z in quet:
        print(f"  {z['Nr']:7d}{z['luot_xuoi']*1e3:12.3f}{z['phan_du']*1e3:15.3f}"
              f"{z['buoc_day_du']*1e3:12.3f}{z['boi_so_phan_du']:10.2f}"
              f"{z['c_menh_de']:11.3f}")
    c = quet[-1]
    print(f"  So bao cao (Nr = {c['Nr']}, luot xuoi du dai de do tin cay):")
    print(f"    do thi phan du / do thi xuoi = {c['boi_so_phan_du']:.2f}"
          f"   (Nhan xet nx:chi-phi-bac-cao uoc luong 4-6 lan)")
    print(f"    c = cost(grad J)/cost(J)     = {c['c_menh_de']:.2f}"
          f"   (Menh de md:chi-phi-bp doi hoi c <= 3)")

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
    out = {"mot_buoc": c, "quet_kich_thuoc": quet, "fd": rows,
           "eps_pinn": eps_pinn,
           "giay_ref_min": t_min, "giay_huan_luyen_pinn": t_pinn}
    if dat:
        r = dat[0]
        print(f"\n  Voi dt = 5e-5 co dinh: FD dat cung muc sai so tai "
              f"Nx = {r['Nx']} trong {r['giay']:.3f} s")
        print(f"  -> PINN cham hon {t_pinn/r['giay']:.0f} lan.")
        out["fd_dat"] = r
        out["cham_hon_lan_dt_co_dinh"] = t_pinn / r["giay"]

        print("\n=== 5. Nhung dt = 5e-5 la lang phi tren luoi tho ===")
        q = chi_phi_fd_dt_hop_ly(r["Nx"], ref_min)
        print(f"  Tai Nx = {r['Nx']}: chan on dinh dt <= "
              f"{q['chan_khuech_tan']:.3e} (khuech tan), "
              f"{q['chan_doi_luu']:.3e} (doi luu)")
        print(f"  {'dt':>10}{'so buoc':>10}{'giay':>10}{'eps_L2':>13}")
        for z in q["lan_chay"]:
            print(f"  {z['dt']:10.1e}{z['n_buoc']:10d}{z['giay']:10.4f}"
                  f"{z['eps_L2']:13.4e}")
        out["fd_quet_dt"] = q
        hop_le = [z for z in q["lan_chay"] if z["eps_L2"] <= eps_pinn]
        if hop_le:
            z = min(hop_le, key=lambda w: w["giay"])
            print(f"\n  Sai so KHONG DOI khi dt giam -- sai so khong gian chiem"
                  f" uu the.")
            print(f"  Chi phi FD that su tai Nx = {r['Nx']}: {z['giay']:.4f} s "
                  f"(dt = {z['dt']:.0e}, {z['n_buoc']} buoc).")
            print(f"  PINN can {t_pinn:.1f} s  ->  cham hon "
                  f"{t_pinn/z['giay']:.0f} lan.")
            out["fd_nhanh_nhat"] = z
            out["cham_hon_lan"] = t_pinn / z["giay"]
    json.dump(out, open(os.path.join(OUT, "exp9_chiphi.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
