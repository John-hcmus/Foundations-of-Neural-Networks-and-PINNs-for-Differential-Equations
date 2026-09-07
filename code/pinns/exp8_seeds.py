"""TN8 --- Do do phan tan giua cac hat giong.

Han che thu nhat ma luan van tu neu o Muc sec:hanche: moi so lieu deu den tu
MOT hat giong duy nhat, nen khong co uoc luong do phan tan, va "cac chenh lech
duoi hai lan khong nen duoc coi la co y nghia".

Thi nghiem nay lam dung viec ma han che ay de nghi: lap lai moi cau hinh tren
nam hat giong roi bao cao TRUNG VI cung KHOANG PHAN VI, thay vi mot gia tri.

Chay lai TN4 (khuech tan) va TN5 (Burgers) voi seed = 0..4.
"""
from __future__ import annotations
import json, os, statistics as st

from . import exp4_heat, exp5_burgers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]


def tom_tat(xs):
    xs = sorted(xs)
    return {"trung_vi": st.median(xs), "min": xs[0], "max": xs[-1],
            "q1": xs[len(xs) // 4], "q3": xs[(3 * len(xs)) // 4], "n": len(xs)}


def dong(ten, s):
    return (f"{ten:<28}{s['trung_vi']:12.4e}{s['min']:12.4e}{s['max']:12.4e}"
            f"{s['max']/s['min']:10.1f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}

    print("=== TN4: phuong trinh khuech tan, 5 hat giong ===")
    heat = {"co_dinh": [], "u_trong_so": []}
    for s in SEEDS:
        heat["co_dinh"].append(exp4_heat.run(False, seed=s))
        heat["u_trong_so"].append(exp4_heat.run(True, seed=s))
    print(f"{'chien luoc':<28}{'trung vi':>12}{'min':>12}{'max':>12}{'max/min':>10}")
    for k, ten in [("co_dinh", "Trong so co dinh"), ("u_trong_so", "U trong so")]:
        s = tom_tat([r["eps_L2_cuoi"] for r in heat[k]])
        out[f"tn4_{k}"] = s
        print(dong(ten, s))
    rho = tom_tat([r["rho_cuoi"] / r["rho_dau"] for r in heat["co_dinh"]])
    out["tn4_ty_so_rho"] = rho
    n_giam = sum(1 for r in heat["co_dinh"] if r["rho_cuoi"] < r["rho_dau"])
    print(f"  rho_b cuoi/dau: trung vi {rho['trung_vi']:.3f} "
          f"[{rho['min']:.3f}; {rho['max']:.3f}]  -> GIAM o {n_giam}/{len(SEEDS)} hat giong")
    ty = tom_tat([b["eps_L2_cuoi"] / a["eps_L2_cuoi"]
                  for a, b in zip(heat["co_dinh"], heat["u_trong_so"])])
    out["tn4_u_lam_xau"] = ty
    print(f"  u trong so lam xau: trung vi {ty['trung_vi']:.1f} lan "
          f"[{ty['min']:.1f}; {ty['max']:.1f}]")

    print("\n=== TN5: Burgers, 5 hat giong ===")
    ref = exp5_burgers.reference(Nx=2047)
    burg = {"co_dinh": [], "u_trong_so": []}
    for s in SEEDS:
        burg["co_dinh"].append(exp5_burgers.run(None, seed=s, ref=ref))
        burg["u_trong_so"].append(exp5_burgers.run(0.1, seed=s, ref=ref))
    print(f"{'chien luoc':<28}{'trung vi':>12}{'min':>12}{'max':>12}{'max/min':>10}")
    for k, ten in [("co_dinh", "Trong so co dinh"), ("u_trong_so", "U trong so a=0,1")]:
        s = tom_tat([r["eps_L2_cuoi"] for r in burg[k]])
        out[f"tn5_{k}"] = s
        print(dong(ten, s))
    rho = tom_tat([r["rho_cuoi"] / r["rho_dau"] for r in burg["co_dinh"]])
    out["tn5_ty_so_rho"] = rho
    n_tang = sum(1 for r in burg["co_dinh"] if r["rho_cuoi"] > r["rho_dau"])
    print(f"  rho_b cuoi/dau: trung vi {rho['trung_vi']:.1f} "
          f"[{rho['min']:.1f}; {rho['max']:.1f}]  -> TANG o {n_tang}/{len(SEEDS)} hat giong")
    ty = tom_tat([b["eps_L2_cuoi"] / a["eps_L2_cuoi"]
                  for a, b in zip(burg["co_dinh"], burg["u_trong_so"])])
    out["tn5_u_lam_xau"] = ty
    print(f"  u trong so lam xau: trung vi {ty['trung_vi']:.1f} lan "
          f"[{ty['min']:.1f}; {ty['max']:.1f}]")

    out["chi_tiet"] = {
        "tn4": {k: [{kk: vv for kk, vv in r.items() if kk != "rho_hist"}
                    for r in v] for k, v in heat.items()},
        "tn5": burg,
    }
    json.dump(out, open(os.path.join(OUT, "exp8_seeds.json"), "w"), indent=2)
    print("\nDa ghi results/exp8_seeds.json")


if __name__ == "__main__":
    main()
