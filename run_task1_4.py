"""Task 1.4: validacion del simulador corregido contra la teoria (Task 1.1) con replicas independientes."""
import json

import numpy as np

import icu_sim as s
import teoria as th

LAM, MU, C = th.LAM, th.MU, th.C
NREP, H, W = 50, 5000.0, 100.0
res = {"config": dict(nrep=NREP, horizon=H, warmup=W)}

mm = th.mmc(LAM, MU, C)
ml = th.mmcc(LAM, MU, C)
teo = {
    "wait": {"L": mm["L"], "Lq": mm["Lq"], "Wq": mm["Wq"], "W": mm["W"], "util": mm["rho"],
             "P_wait": mm["C"], "P_block": 0.0, "lam_ef": LAM},
    "loss": {"L": ml["L"], "Lq": 0.0, "Wq": 0.0, "W": 1 / MU, "util": ml["util"],
             "P_wait": 0.0, "P_block": ml["B"], "lam_ef": ml["lam_ef"]},
}

for var, wait, seed in (("wait", True, 14), ("loss", False, 15)):
    runs = s.run_replicas(NREP, seed, lam=LAM, mu=MU, c=C, horizon=H, warmup=W, wait=wait)
    summ = s.summarize(runs)
    res[var] = {}
    print(f"=== variante {var} ===")
    for k, (m, lo, hi) in summ.items():
        t = teo[var][k]
        inside = lo <= t <= hi
        # medidas identicamente cero en la teoria (p.ej. Lq en la variante sin espera): IC degenerado
        print(f"  {k:8s} media={m:.5f}  IC95%=[{lo:.5f}, {hi:.5f}]  teoria={t:.5f}  dentro={inside}")
        res[var][k] = dict(media=m, lo=lo, hi=hi, teoria=t, dentro=bool(inside))
    # Little con las tres cantidades de la simulacion (independiente de la teoria), por replica
    L = np.array([r["L"] for r in runs])
    lam_ef = np.array([r["lam_ef"] for r in runs])
    Wm = np.array([r["W"] for r in runs])
    err = np.abs(L - lam_ef * Wm) / L
    Lq = np.array([r["Lq"] for r in runs])
    Wq = np.array([r["Wq"] for r in runs])
    errq = np.abs(Lq - lam_ef * Wq) / np.where(Lq > 0, Lq, np.nan)
    print(f"  Little L = lam_ef W : L={L.mean():.4f} lam_ef*W={np.mean(lam_ef * Wm):.4f} "
          f"err rel (medias)={abs(L.mean() - np.mean(lam_ef) * Wm.mean()) / L.mean() * 100:.3f}%  "
          f"err rel medio por replica={err.mean() * 100:.3f}% (max {err.max() * 100:.3f}%)")
    little = dict(L=L.mean(), lam_ef=lam_ef.mean(), W=Wm.mean(),
                  lam_ef_W=float(np.mean(lam_ef) * Wm.mean()),
                  err_rel_medias=float(abs(L.mean() - lam_ef.mean() * Wm.mean()) / L.mean()),
                  err_rel_rep_medio=float(err.mean()), err_rel_rep_max=float(err.max()))
    if wait:
        print(f"  Little Lq = lam_ef Wq : Lq={Lq.mean():.4f} lam_ef*Wq={np.mean(lam_ef) * Wq.mean():.4f} "
              f"err rel (medias)={abs(Lq.mean() - lam_ef.mean() * Wq.mean()) / Lq.mean() * 100:.3f}%  "
              f"err rel medio por replica={np.nanmean(errq) * 100:.3f}%")
        little.update(Lq=Lq.mean(), Wq=Wq.mean(), lam_ef_Wq=float(lam_ef.mean() * Wq.mean()),
                      errq_rel_medias=float(abs(Lq.mean() - lam_ef.mean() * Wq.mean()) / Lq.mean()),
                      errq_rel_rep_medio=float(np.nanmean(errq)))
    else:
        # error de usar lambda en lugar de lambda_ef
        little["lam_W"] = float(LAM * Wm.mean())
        little["err_rel_lambda"] = float(abs(L.mean() - LAM * Wm.mean()) / L.mean())
        print(f"  (con lambda en vez de lam_ef: lam*W={LAM * Wm.mean():.4f}, err rel {little['err_rel_lambda'] * 100:.2f}%)")
    res[var]["little"] = little

json.dump(res, open("resultados/task1_4.json", "w"), indent=1, default=float)
