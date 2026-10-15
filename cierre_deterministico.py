"""Task 2.3c (descomposicion): cierre con llegadas Y estancia deterministas (D/D/17). Sin aleatoriedad:
si el modelo de acumulacion fuera la causa, aqui deberia cumplirse. Una sola trayectoria basta."""
import json
import numpy as np
import icu_sim as s
import teoria as th

LAM, MU, C, KOFF, TB = th.LAM, th.MU, th.C, th.K_OFF, th.T_B
T0, HOR, STEP = 150.0, 150.0 + 7.0 + 60.0, 0.05
grid = np.arange(T0 - 5.0, HOR - 1.0, STEP)
rng = np.random.default_rng(0)
r = s.simulate(LAM, MU, C, HOR, 0.0, rng, wait=True, service="det", closure=(T0, TB, KOFF), grid=grid,
               interarrival=lambda _r: 1.0 / LAM)
Q, N = r["grid_Q"], r["grid_N"]
g = grid - T0 - TB
pre = (grid < T0)
print("antes del cierre: Q max =", Q[pre].max(), " N en [", N[pre].min(), ",", N[pre].max(), "]")
print("Q pico (fin del cierre) =", Q[(g > -0.2) & (g < 0)].max())
after = g >= 0
i_empty = np.where(after & (Q == 0))[0][0]
print("primer instante con Q=0 tras reabrir:", round(g[i_empty], 2), "d")
print("N tras reabrir: ", {t: float(N[np.argmin(abs(g - t))]) for t in (0.1, 1, 2, 3, 4, 6, 8, 12)})
print("Q tras reabrir: ", {t: float(Q[np.argmin(abs(g - t))]) for t in (0.1, 1, 2, 3, 4, 6, 8, 12)})
out = dict(Q_pre_max=float(Q[pre].max()), Q_pico=float(Q[(g > -0.2) & (g < 0)].max()), t_Q0=float(g[i_empty]))
json.dump(out, open("resultados/cierre_deterministico.json", "w"), indent=1)

# ultimo instante con fila > 0 tras reabrir (fila vacia de forma permanente)
idx = np.where(after & (Q > 0))[0]
t_last = float(g[idx[-1]]) if len(idx) else 0.0
print("ultimo instante con Q>0 tras reabrir (D/D):", round(t_last, 2), "d")
out["t_last_Qpos"] = t_last

# dispersion de la fila en el instante de reapertura (estancia exponencial, llegadas Poisson)
gp = np.array([T0 + TB - 1e-6])
qs, ns = [], []
for rg in s.make_rngs(77, 4000):
    x = s.simulate(LAM, MU, C, T0 + TB + 1.0, 0.0, rg, wait=True, service="exp", closure=(T0, TB, KOFF), grid=gp)
    qs.append(x["grid_Q"][0]); ns.append(x["grid_N"][0])
qs, ns = np.array(qs), np.array(ns)
busy = ns - qs
libres = C - busy
exceso = np.maximum(qs - libres, 0)
print(f"fila al reabrir (exp, 4000 rep): media={qs.mean():.2f} sd={qs.std(ddof=1):.2f} min={qs.min():.0f} max={qs.max():.0f} "
      f"camas ocupadas={busy.mean():.2f}  E[(Q - libres)+]={exceso.mean():.2f}  P(Q<=libres)={np.mean(qs <= libres):.3f}")
out.update(Q_reap_media=float(qs.mean()), Q_reap_sd=float(qs.std(ddof=1)), busy_reap=float(busy.mean()),
           exceso_medio=float(exceso.mean()), p_vacia_al_reabrir=float(np.mean(qs <= libres)))
json.dump(out, open("resultados/cierre_deterministico.json", "w"), indent=1)
