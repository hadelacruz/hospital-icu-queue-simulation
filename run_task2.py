"""Tasks 2.2, 2.3 y 2.4: estancia no exponencial, cierre parcial de camas y estancia +25 %.
Las predicciones (resultados/predicciones_task2_1.md) se escribieron ANTES de ejecutar este script."""
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import icu_sim as s
import teoria as th

LAM, MU, C, KOFF, TB = th.LAM, th.MU, th.C, th.K_OFF, th.T_B
mm = th.mmc(LAM, MU, C)
res = {}

# ------------------------------------------------------------------ Task 2.2
NREP22, H22, W22 = 100, 10000.0, 200.0
print("=== Task 2.2: Wq con tres distribuciones de estancia (media 1/mu) ===")
res["t22"] = {}
for name, dist, cs in (("determinista", "det", 0.0), ("exponencial", "exp", 1.0), ("lognormal", "lognormal", 2.0)):
    runs = s.run_replicas(NREP22, 22, lam=LAM, mu=MU, c=C, horizon=H22, warmup=W22, wait=True, service=dist, cs=cs)
    summ = s.summarize(runs)
    ac = mm["Wq"] * 24 * (1 + cs ** 2) / 2          # Allen-Cunneen, en horas
    wq, lo, hi = (x * 24 for x in summ["Wq"])
    rel = (ac - wq) / wq
    res["t22"][name] = dict(cs=cs, Wq_h=wq, lo_h=lo, hi_h=hi, AC_h=ac, err_rel=rel,
                            AC_dentro=bool(lo <= ac <= hi), Lq=summ["Lq"][0], Lq_lo=summ["Lq"][1], Lq_hi=summ["Lq"][2],
                            P_wait=summ["P_wait"][0], util=summ["util"][0],
                            L=summ["L"][0], W_h=summ["W"][0] * 24)
    print(f"  {name:13s} Cs={cs}: Wq_sim={wq:.3f} h [{lo:.3f},{hi:.3f}]  AC={ac:.3f} h  err_rel(AC vs sim)={rel * 100:+.1f}%  "
          f"Lq={summ['Lq'][0]:.4f} P_wait={summ['P_wait'][0]:.4f}")

# ------------------------------------------------------------------ Task 2.4a
print("\n=== Task 2.4a: M/M/c/c con estancia media +25 % (mu' = 0.2) ===")
mu2 = MU / 1.25
runs = s.run_replicas(30, 24, lam=LAM, mu=mu2, c=C, horizon=5000.0, warmup=100.0, wait=False)
b, blo, bhi = s.ci95([r["P_block"] for r in runs])
b_th = th.erlang_b(C, LAM / mu2)
b0_th = th.erlang_b(C, th.A)
runs0 = s.run_replicas(30, 14, lam=LAM, mu=MU, c=C, horizon=5000.0, warmup=100.0, wait=False)
b0, b0lo, b0hi = s.ci95([r["P_block"] for r in runs0])
print(f"  B_sim(+25%) = {b:.5f} [{blo:.5f},{bhi:.5f}]  B_teoria = {b_th:.5f}  (prediccion 2.1c: 0.1416)")
print(f"  B_sim(base) = {b0:.5f} [{b0lo:.5f},{b0hi:.5f}]  B_teoria = {b0_th:.5f}")
print(f"  razon simulada = {b / b0:.3f}  (prediccion 2.47)")
res["t24a"] = dict(B=b, lo=blo, hi=bhi, B_teo=b_th, B0=b0, B0_lo=b0lo, B0_hi=b0hi, B0_teo=b0_th,
                   razon_sim=b / b0, razon_teo=b_th / b0_th, util=float(np.mean([r["util"] for r in runs])),
                   lam_ef=float(np.mean([r["lam_ef"] for r in runs])))

# ------------------------------------------------------------------ Task 2.3
T0 = 150.0                      # instante del cierre (> calentamiento de 100 d justificado con Welch)
HOR = T0 + TB + 150.0
STEP = 0.25
grid = np.arange(T0 - 20.0, HOR - 1.0, STEP)
NREP23 = 6000


def closure_run(dist, seed, nrep):
    runs = [s.simulate(LAM, MU, C, HOR, 0.0, r, wait=True, service=dist, closure=(T0, TB, KOFF), grid=grid)
            for r in s.make_rngs(seed, nrep)]
    return np.array([x["grid_Q"] for x in runs]), np.array([x["grid_N"] for x in runs])


def recovery(meanQ, thr):
    after = grid >= T0 + TB
    idx = np.where(after & (meanQ < thr))[0]
    return None if len(idx) == 0 else float(grid[idx[0]] - (T0 + TB))


lq_ss = {"exp": mm["Lq"], "det": res["t22"]["determinista"]["Lq"]}
print(f"\n=== Task 2.3: cierre de {KOFF} camas durante {TB:g} d (capacidad {C - KOFF}); {NREP23} replicas ===")
res["t23"] = {}
curves = {}
rng_boot = np.random.default_rng(5)
for dist in ("exp", "det"):
    Q, N = closure_run(dist, 23 if dist == "exp" else 230, NREP23)
    mq = Q.mean(axis=0)
    curves[dist] = (mq, N.mean(axis=0))
    thr = 1.1 * lq_ss[dist]
    trec = recovery(mq, thr)
    trec100 = recovery(Q[:100].mean(axis=0), thr)
    boots = []
    for _ in range(300):
        ii = rng_boot.integers(0, NREP23, NREP23)
        v = recovery(Q[ii].mean(axis=0), thr)
        boots.append(np.nan if v is None else v)
    boots = np.array(boots, dtype=float)
    ipk = int(np.argmax(mq))
    i_re = int(np.argmin(np.abs(grid - (T0 + TB))))
    res["t23"][dist] = dict(trec=trec, trec_100=trec100, boot_lo=float(np.nanpercentile(boots, 2.5)),
                            boot_hi=float(np.nanpercentile(boots, 97.5)), Lq_ss=lq_ss[dist], thr=thr,
                            Q_pico=float(mq[ipk]), t_pico=float(grid[ipk] - T0), Q_reapertura=float(mq[i_re]),
                            N_reapertura=float(curves[dist][1][i_re]),
                            Q_prev=float(mq[(grid >= T0 - 20) & (grid < T0)].mean()))
    print(f"  {dist}: Lq_ss={lq_ss[dist]:.4f} umbral={thr:.4f} T_rec={trec} d (100 rep: {trec100}) "
          f"IC boot=[{res['t23'][dist]['boot_lo']:.2f},{res['t23'][dist]['boot_hi']:.2f}]  pico={mq[ipk]:.2f} en t0+{grid[ipk] - T0:.1f}  "
          f"Q(reapertura)={mq[i_re]:.2f}  N(reapertura)={curves[dist][1][i_re]:.2f}")

# formula del Task 2.1b y refinamiento POST-HOC (no es prediccion): cierre "gradual" + camas liberadas absorben la fila
r_up = LAM - (C - KOFF) * MU
r_dn = C * MU - LAM
Td = r_up * TB / r_dn
t1 = math.log(th.A / (C - KOFF)) / MU            # tiempo hasta que los pacientes en cama bajan de 12.8 a 9
q_pico_ref = LAM * t1 + r_up * (TB - t1)
td_ref = max(0.0, q_pico_ref - KOFF) / r_dn
res["formula"] = dict(Td=Td, Q_pico=r_up * TB, r_up=r_up, r_dn=r_dn, t1=t1, Q_pico_ref=q_pico_ref, Td_ref=td_ref)
print(f"\n  formula 2.1b: Td={Td:.3f} d, Q_pico={r_up * TB:.2f};  refinado post-hoc: t1={t1:.2f} d, Q_pico={q_pico_ref:.2f}, Td={td_ref:.2f} d")

json.dump(res, open("resultados/task2.json", "w"), indent=1, default=float)
np.savez("resultados/curvas_cierre.npz", grid=grid, exp_Q=curves["exp"][0], exp_N=curves["exp"][1],
         det_Q=curves["det"][0], det_N=curves["det"][1])

# ------------------------------------------------------------------ figura 2.3b
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = grid - T0
for a, xlim in ((ax[0], (-20, TB + 100)), (ax[1], (-2, TB + 15))):
    a.axvspan(0, TB, color="#f4c7c3", alpha=0.6, label=f"cierre ({KOFF} camas, {TB:g} d)")
    a.plot(x, curves["exp"][0], color="#1f5fa8", lw=1.6, label="estancia exponencial")
    a.plot(x, curves["det"][0], color="#2e8b57", lw=1.6, label="estancia determinista")
    a.axhline(1.1 * lq_ss["exp"], color="#1f5fa8", ls="--", lw=0.9)
    a.axhline(1.1 * lq_ss["det"], color="#2e8b57", ls="--", lw=0.9)
    a.axvline(TB + Td, color="k", ls=":", lw=1.2, label=f"fórmula 2.1b: $T_d$={Td:.2f} d tras reabrir")
    a.set_xlim(*xlim)
    a.set_xlabel("días desde el inicio del cierre"); a.set_ylabel(r"$L_q(t)$ promedio entre réplicas")
ax[0].set_title(f"Fila promedio ({NREP23} réplicas)")
ax[1].set_title("Detalle del cierre y la recuperación")
ax[0].legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("figuras/recuperacion.pdf"); plt.savefig("figuras/recuperacion.png", dpi=150)
