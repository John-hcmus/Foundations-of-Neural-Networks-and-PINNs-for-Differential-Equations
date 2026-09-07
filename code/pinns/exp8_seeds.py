"""TN8 --- Do do phan tan giua cac hat giong.

Han che thu nhat ma luan van tu neu o Muc sec:hanche: moi so lieu deu den tu
MOT hat giong duy nhat, nen khong co uoc luong do phan tan, va "cac chenh lech
duoi hai lan khong nen duoc coi la co y nghia".

Thi nghiem nay lam dung viec ma han che ay de nghi: lap moi cau hinh tren nam
hat giong roi bao cao TRUNG VI cung KHOANG, thay vi mot gia tri.

Chay:
    python -m pinns.exp8_seeds heat       # chi TN4
    python -m pinns.exp8_seeds burgers    # chi TN5
    python -m pinns.exp8_seeds pho        # chi TN3
    python -m pinns.exp8_seeds lambda     # chi TN2
    python -m pinns.exp8_seeds            # ca bon

Ket qua duoc ghi ra results/ SAU MOI HAT GIONG, va mot lan chay moi se DOC LAI
phan da co roi chi chay tiep nhung hat giong con thieu. Nho vay mot lan chay bi
ngat khong lam mat gi.
"""
from __future__ import annotations
import json, os, statistics as st, sys, time

import numpy as np

from . import exp2_lambda, exp3_spectral, exp4_heat, exp5_burgers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]


def tom_tat(xs):
    xs = sorted(xs)
    return {"trung_vi": st.median(xs), "min": xs[0], "max": xs[-1], "n": len(xs)}


def _doc(ten):
    """Doc lai ket qua da co, de chay tiep thay vi chay lai tu dau."""
    f = os.path.join(OUT, f"exp8_{ten}.json")
    if not os.path.exists(f):
        return [], []
    with open(f) as fh:
        d = json.load(fh)
    return d.get("co_dinh", []), d.get("u_trong_so", [])


def _ghi(ten, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"exp8_{ten}.json"), "w") as fh:
        json.dump(data, fh, indent=2)


def _bang(ten_bai, co_dinh, u_ts, khoa_rho, chieu):
    print(f"\n--- {ten_bai}: tong ket tren {len(co_dinh)} hat giong ---")
    print(f"{'chien luoc':<22}{'trung vi':>12}{'min':>12}{'max':>12}{'max/min':>10}")
    out = {}
    for ten, rs in [("Trong so co dinh", co_dinh), ("U trong so", u_ts)]:
        s = tom_tat([r["eps_L2_cuoi"] for r in rs])
        out[ten] = s
        print(f"{ten:<22}{s['trung_vi']:12.4e}{s['min']:12.4e}"
              f"{s['max']:12.4e}{s['max']/s['min']:10.1f}")
    ty = tom_tat([b["eps_L2_cuoi"] / a["eps_L2_cuoi"] for a, b in zip(co_dinh, u_ts)])
    out["u_lam_xau"] = ty
    print(f"  u trong so lam xau: trung vi {ty['trung_vi']:.1f} lan "
          f"[{ty['min']:.1f}; {ty['max']:.1f}]")
    rho = tom_tat([r["rho_cuoi"] / r["rho_dau"] for r in co_dinh])
    out["ty_so_rho"] = rho
    if chieu == "giam":
        n = sum(1 for r in co_dinh if r["rho_cuoi"] < r["rho_dau"])
    else:
        n = sum(1 for r in co_dinh if r["rho_cuoi"] > r["rho_dau"])
    out["so_hat_giong_dung_chieu"] = n
    print(f"  rho_b cuoi/dau: trung vi {rho['trung_vi']:.3f} "
          f"[{rho['min']:.3f}; {rho['max']:.3f}]"
          f"  -> {chieu.upper()} o {n}/{len(co_dinh)} hat giong")
    return out


def chay_heat():
    co_dinh, u_ts = _doc("heat")
    xong = min(len(co_dinh), len(u_ts))
    co_dinh, u_ts = co_dinh[:xong], u_ts[:xong]
    if xong:
        print(f"[TN4] da co {xong} hat giong, chay tiep tu hat giong {SEEDS[xong]}",
              flush=True)
    for s in SEEDS[xong:]:
        t0 = time.time()
        co_dinh.append(exp4_heat.run(False, seed=s))
        u_ts.append(exp4_heat.run(True, seed=s))
        print(f"[TN4] hat giong {s}: co dinh {co_dinh[-1]['eps_L2_cuoi']:.4e}, "
              f"u {u_ts[-1]['eps_L2_cuoi']:.4e}  ({time.time()-t0:.0f}s)", flush=True)
        _ghi("heat", {"co_dinh": [{k: v for k, v in r.items() if k != "rho_hist"}
                                  for r in co_dinh],
                      "u_trong_so": [{k: v for k, v in r.items() if k != "rho_hist"}
                                     for r in u_ts]})
    return _bang("TN4 khuech tan", co_dinh, u_ts, "rho", "giam")


def chay_burgers():
    co_dinh, u_ts = _doc("burgers")
    xong = min(len(co_dinh), len(u_ts))
    co_dinh, u_ts = co_dinh[:xong], u_ts[:xong]
    if xong == len(SEEDS):
        return _bang("TN5 Burgers", co_dinh, u_ts, "rho", "tang")
    print("Dang tinh nghiem tham chieu ...", flush=True)
    ref = exp5_burgers.reference(Nx=2047)
    if xong:
        print(f"[TN5] da co {xong} hat giong, chay tiep tu hat giong {SEEDS[xong]}",
              flush=True)
    for s in SEEDS[xong:]:
        t0 = time.time()
        co_dinh.append(exp5_burgers.run(None, seed=s, ref=ref))
        u_ts.append(exp5_burgers.run(0.1, seed=s, ref=ref))
        print(f"[TN5] hat giong {s}: co dinh {co_dinh[-1]['eps_L2_cuoi']:.4e}, "
              f"u {u_ts[-1]['eps_L2_cuoi']:.4e}  ({time.time()-t0:.0f}s)", flush=True)
        _ghi("burgers", {"co_dinh": co_dinh, "u_trong_so": u_ts})
    return _bang("TN5 Burgers", co_dinh, u_ts, "rho", "tang")


def _doc_pho():
    f = os.path.join(OUT, "exp8_pho.json")
    if not os.path.exists(f):
        return []
    with open(f) as fh:
        return json.load(fh).get("theo_hat_giong", [])


def chay_pho():
    """TN3 tren nam hat giong.

    TN3 dung LUOI TAT DINH (uniform_1d), nen nguon ngau nhien duy nhat la
    khoi tao mang. Ba khang dinh cua Muc sec:tn3 can kiem chung:
      (1) che do hoi quy hoc theo THU TU TANG DAN cua k (thien kien pho);
      (2) che do phan du DAO NGUOC thu tu ay, do luong bang |c_1|/|c_8| > 1;
      (3) dac trung Fourier cai thien sai so mot cach dang ke.
    """
    xong = _doc_pho()
    if xong:
        print(f"[TN3] da co {len(xong)} hat giong, chay tiep tu hat giong "
              f"{SEEDS[len(xong)]}", flush=True)
    for s in SEEDS[len(xong):]:
        t0 = time.time()
        r = {"seed": s}
        for m in ("regression", "residual", "fourier"):
            z = exp3_spectral.run(m, seed=s)
            # khoa phai la CHUOI: sau khi ghi/doc lai JSON thi khoa so bien
            # thanh chuoi, va vong chay tiep se doc nham neu ta dung khoa so.
            r[m] = {"t10": {str(k): v for k, v in z["t10"].items()},
                    "c_cuoi": {str(k): v for k, v in z["c_cuoi"].items()},
                    "eps_L2": z["eps_L2"]}
        xong.append(r)
        print(f"[TN3] hat giong {s}: hoi quy t10 = "
              f"{[r['regression']['t10'][str(k)] for k in exp3_spectral.KS]}, "
              f"phan du |c1|/|c8| = "
              f"{r['residual']['c_cuoi']['1'] / r['residual']['c_cuoi']['8']:.1f}, "
              f"Fourier cai thien "
              f"{r['residual']['eps_L2'] / r['fourier']['eps_L2']:.0f} lan"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        _ghi("pho", {"theo_hat_giong": xong})
    return _bang_pho(xong)


def _bang_pho(rs):
    KS = exp3_spectral.KS
    n = len(rs)
    print(f"\n--- TN3 thien kien pho: tong ket tren {n} hat giong ---")

    don_dieu = sum(1 for r in rs if _tang_dan(
        [r["regression"]["t10"][str(k)] for k in KS]))
    print(f"  (1) hoi quy, t10 tang dan theo k: {don_dieu}/{n} hat giong")
    for r in rs:
        print(f"      hat giong {r['seed']}: "
              f"{[r['regression']['t10'][str(k)] for k in KS]}")

    ty = tom_tat([r["residual"]["c_cuoi"]["1"] / r["residual"]["c_cuoi"]["8"]
                  for r in rs])
    dao = sum(1 for r in rs
              if r["residual"]["c_cuoi"]["1"] > r["residual"]["c_cuoi"]["8"])
    print(f"  (2) phan du, |c1|/|c8|: trung vi {ty['trung_vi']:.1f} "
          f"[{ty['min']:.1f}; {ty['max']:.1f}]  -> DAO CHIEU o {dao}/{n}")

    ci = tom_tat([r["residual"]["eps_L2"] / r["fourier"]["eps_L2"] for r in rs])
    tot = sum(1 for r in rs if r["fourier"]["eps_L2"] < r["residual"]["eps_L2"])
    print(f"  (3) Fourier cai thien: trung vi {ci['trung_vi']:.0f} lan "
          f"[{ci['min']:.0f}; {ci['max']:.0f}]  -> tot hon o {tot}/{n}")

    eps_f = tom_tat([r["fourier"]["eps_L2"] for r in rs])
    print(f"      eps_L2 cua che do Fourier: trung vi {eps_f['trung_vi']:.4e} "
          f"[{eps_f['min']:.4e}; {eps_f['max']:.4e}]")
    return {"n": n, "hoi_quy_don_dieu": don_dieu, "ty_c1_c8": ty,
            "so_dao_chieu": dao, "cai_thien_fourier": ci, "so_tot_hon": tot,
            "eps_fourier": eps_f,
            "t10_hoi_quy": {str(k): [r["regression"]["t10"][str(k)]
                                     for r in rs] for k in KS}}


def _tang_dan(xs):
    """True neu day tang dan that su, coi None (khong dat nguong) la +vo cung."""
    ys = [float("inf") if v is None else v for v in xs]
    return all(a < b for a, b in zip(ys, ys[1:]))


LAMS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def chay_lambda():
    """TN2 tren nam hat giong.

    Hai khang dinh cua Muc sec:tn2:
      (1) sai so bien ti le lam_b^{-1}, tuc so mu hoi quy log--log gan -1;
      (2) eps_L2 dat cuc tieu tai mot lam_b HUU HAN, khong phai lam_b -> vo cung.
    Ca hai deu duoc rut tu mot hat giong duy nhat.
    """
    f = os.path.join(OUT, "exp8_lambda.json")
    xong = json.load(open(f))["theo_hat_giong"] if os.path.exists(f) else []
    if xong:
        print(f"[TN2] da co {len(xong)} hat giong, chay tiep tu hat giong "
              f"{SEEDS[len(xong)]}", flush=True)
    for sd in SEEDS[len(xong):]:
        t0 = time.time()
        rows = [exp2_lambda.run(l, seed=sd) for l in LAMS]
        sub = [w for w in rows if 0.1 <= w["lam_b"] <= 100.0]
        so_mu = float(np.polyfit(np.log10([w["lam_b"] for w in sub]),
                                 np.log10([w["sai_so_bien"] for w in sub]), 1)[0])
        tot_nhat = min(rows, key=lambda w: w["eps_L2"])["lam_b"]
        xong.append({"seed": sd, "so_mu": so_mu, "lam_tot_nhat": tot_nhat,
                     "rows": rows})
        print(f"[TN2] hat giong {sd}: so mu {so_mu:.3f}, "
              f"lam_b tot nhat {tot_nhat:g}  ({time.time()-t0:.0f}s)", flush=True)
        _ghi("lambda", {"theo_hat_giong": xong})

    n = len(xong)
    print(f"\n--- TN2 rang buoc mem: tong ket tren {n} hat giong ---")
    sm = tom_tat([r["so_mu"] for r in xong])
    print(f"  (1) so mu log--log: trung vi {sm['trung_vi']:.3f} "
          f"[{sm['min']:.3f}; {sm['max']:.3f}]   (du bao -1)")
    print(f"      tung hat giong: "
          f"{', '.join('%.3f' % r['so_mu'] for r in xong)}")
    huu_han = sum(1 for r in xong if r["lam_tot_nhat"] < max(LAMS))
    print(f"  (2) lam_b toi uu HUU HAN o {huu_han}/{n} hat giong")
    print(f"      tung hat giong: "
          f"{', '.join('%g' % r['lam_tot_nhat'] for r in xong)}")
    return {"n": n, "so_mu": sm, "so_huu_han": huu_han,
            "lam_tot_nhat": [r["lam_tot_nhat"] for r in xong]}


def main():
    phan = sys.argv[1] if len(sys.argv) > 1 else "ca_bon"
    if phan in ("heat", "ca_bon"):
        _ghi("heat_tomtat", chay_heat())
    if phan in ("burgers", "ca_bon"):
        _ghi("burgers_tomtat", chay_burgers())
    if phan in ("pho", "ca_bon"):
        _ghi("pho_tomtat", chay_pho())
    if phan in ("lambda", "ca_bon"):
        _ghi("lambda_tomtat", chay_lambda())


if __name__ == "__main__":
    main()
