"""Task 2.3c: diagnostico de la discrepancia entre la formula de vaciado (6.33 d) y el tiempo de recuperacion simulado."""
import json
import math

import numpy as np

import teoria as th

d = np.load("resultados/curvas_cierre.npz")
res = json.load(open("resultados/task2.json"))
g = d["grid"] - 150.0 - th.T_B            # dias desde la reapertura
lq_ss = {"exp": th.mmc(th.LAM, th.MU, th.C)["Lq"], "det": res["t22"]["determinista"]["Lq"]}

out = {}
# parametro de decaimiento de Van Doorn para M/M/c: gamma = c mu (1 - sqrt(rho))^2   (modo mas lento de la cola)
gamma = th.C * th.MU * (1 - math.sqrt(th.RHO)) ** 2
out["gamma_teo"] = gamma
out["tau_teo"] = 1 / gamma
for k in ("exp", "det"):
    q = d[f"{k}_Q"]
    ex = q - lq_ss[k]
    sel = (g >= 12) & (g <= 40) & (ex > 0)
    slope, icpt = np.polyfit(g[sel], np.log(ex[sel]), 1)
    out[f"{k}_gamma_ajuste"] = -slope
    out[f"{k}_tau_ajuste"] = -1 / slope
    # sensibilidad al umbral
    sens = {}
    for fac in (1.1, 1.25, 1.5, 2.0):
        idx = np.where((g >= 0) & (q < fac * lq_ss[k]))[0]
        sens[str(fac)] = float(g[idx[0]]) if len(idx) else None
    out[f"{k}_sens"] = sens
    # instante en que el promedio de la fila cae por debajo de 1 paciente y de 0.5 d (modelo fluido: fila 0)
    out[f"{k}_t_Q_lt_1"] = float(g[np.where((g >= 0) & (q < 1.0))[0][0]])
    # perfil
    out[f"{k}_perfil"] = {str(t): (float(q[np.argmin(abs(g - t))]), float(d[f"{k}_N"][np.argmin(abs(g - t))]))
                          for t in (0.25, 1, 2, 4, 6, 10, 15, 20, 30, 40, 60)}
print(json.dumps(out, indent=1))
json.dump(out, open("resultados/analisis_cierre.json", "w"), indent=1)
