"""TN8 --- Do do phan tan giua cac hat giong.

Han che thu nhat ma luan van tu neu o Muc sec:hanche: moi so lieu deu den tu
MOT hat giong duy nhat, nen khong co uoc luong do phan tan, va "cac chenh lech
duoi hai lan khong nen duoc coi la co y nghia".

Thi nghiem nay lam dung viec ma han che ay de nghi: lap moi cau hinh tren nam
hat giong roi bao cao TRUNG VI cung KHOANG, thay vi mot gia tri.

Chay:
    python -m pinns.exp8_seeds heat       # chi TN4
    python -m pinns.exp8_seeds burgers    # chi TN5
    python -m pinns.exp8_seeds            # ca hai

Ket qua duoc ghi ra results/ SAU MOI HAT GIONG, va mot lan chay moi se DOC LAI
phan da co roi chi chay tiep nhung hat giong con thieu. Nho vay mot lan chay bi
ngat khong lam mat gi.
"""
from __future__ import annotations
import json, os, statistics as st, sys, time

from . import exp4_heat, exp5_burgers

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


def main():
    phan = sys.argv[1] if len(sys.argv) > 1 else "ca_hai"
    if phan in ("heat", "ca_hai"):
        _ghi("heat_tomtat", chay_heat())
    if phan in ("burgers", "ca_hai"):
        _ghi("burgers_tomtat", chay_burgers())


if __name__ == "__main__":
    main()
