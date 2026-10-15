"""Bateria de pruebas del Task 1.3 / 1.4: casos con respuesta conocida a mano.
Se ejecuta contra el codigo ORIGINAL (anexo/simulador_original.py) y el CORREGIDO (icu_sim.py)."""
import importlib.util
import json
import types

import numpy as np

import icu_sim as new
import teoria as th

spec = importlib.util.spec_from_file_location("orig", "anexo/simulador_original.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

BIG = 1e9


class Scripted:
    """RNG falso para el original: expovariate(rate) devuelve valores prefijados."""

    def __init__(self, lam, mu, arr, srv):
        self.lam, self.mu = lam, mu
        self.arr, self.srv = iter(arr), iter(srv)

    def expovariate(self, rate):
        return next(self.arr, BIG) if rate == self.lam else next(self.srv, BIG)


def run_orig(arr, srv, c, H, W, wait=True):
    lam, mu = 1.0, 2.0
    real_random = orig.random
    orig.random = types.SimpleNamespace(Random=lambda seed: Scripted(lam, mu, arr, srv))
    try:
        return orig.simulate(lam, mu, c, H, W, 0, wait)
    finally:
        orig.random = real_random  # restaurar el RNG real para las pruebas estadisticas


def run_new(arr, srv, c, H, W, wait=True, closure=None):
    a, s = iter(arr), iter(srv)
    return new.simulate(1.0, 2.0, c, H, W, None, wait=wait, closure=closure,
                        interarrival=lambda r: next(a, BIG), service_fn=lambda r: next(s, BIG))


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol


results = []


def check(nombre, esperado, got_o, got_n, keys):
    row = {"prueba": nombre, "esperado": esperado}
    for tag, got in (("original", got_o), ("corregido", got_n)):
        if got is None:
            row[tag] = None
            continue
        ok = all(close(got[k], v) for k, v in esperado.items())
        row[tag] = {"ok": ok, "valores": {k: got[k] for k in keys}}
    results.append(row)

    def fmt(g):
        if g is None:
            return "N/A"
        return ("OK " if g["ok"] else "FALLA") + " " + str({k: round(v, 4) for k, v in g["valores"].items()})

    print(f"[{nombre}] esperado={ {k: round(v, 4) for k, v in esperado.items()} }")
    print(f"    original : {fmt(row['original'])}")
    print(f"    corregido: {fmt(row['corregido'])}")


# T1: un paciente llega en t=2, estancia 3, c=1, H=10, W=0  ->  L = util = 3/10
e = {"L": 0.3, "util": 0.3, "Lq": 0.0, "P_block": 0.0}
check("T1 un paciente (L ponderado por tiempo)", e,
      run_orig([2, BIG], [3], 1, 10, 0), run_new([2, BIG], [3], 1, 10, 0), list(e))
# T2: igual pero calentamiento 4 -> observa (4,10]: L = (5-4)/6
e = {"L": 1 / 6, "util": 1 / 6}
check("T2 calentamiento: solo (4,10]", e,
      run_orig([2, BIG], [3], 1, 10, 4), run_new([2, BIG], [3], 1, 10, 4), list(e))
# T3: llega en t=8, estancia 5 (sale en 13 > H=10): L = 2/10 (tramo final recortado en H)
e = {"L": 0.2, "util": 0.2}
check("T3 paciente dentro al terminar el horizonte", e,
      run_orig([8, BIG], [5], 1, 10, 0), run_new([8, BIG], [5], 1, 10, 0), list(e))
# T4: sin espera, 1 cama, llegadas t=1,2,3, estancia 1.5 -> bloquea la 2a: P_block=1/3, L=0.3
e = {"P_block": 1 / 3, "L": 0.3}
check("T4 M/M/1/1 determinista (bloqueo)", e,
      run_orig([1, 1, 1, BIG], [1.5, 1.5], 1, 10, 0, wait=False),
      run_new([1, 1, 1, BIG], [1.5, 1.5], 1, 10, 0, wait=False), list(e))
# T5: D/D/1: llegadas t=1..10, estancia 1.5, c=1, H=10. A mano: Wq=2.25, Lq=1.35, L=2.25, util=0.9, P_wait=0.9
e = {"Wq": 2.25, "Lq": 1.35, "L": 2.25, "util": 0.9, "P_wait": 0.9}
check("T5 D/D/1 (una cama)", e,
      run_orig([1.0] * 10 + [BIG], [1.5] * 10, 1, 10, 0),
      run_new([1.0] * 10 + [BIG], [1.5] * 10, 1, 10, 0), list(e))
# T6: cierre total t in [2,5), llega 1 paciente en t=3, estancia 1: espera 2 d; Lq=0.2, L=0.3
e = {"Wq": 2.0, "W": 3.0, "Lq": 0.2, "L": 0.3}
check("T6 cierre parcial (c=1, k_off=1)", e, None,
      run_new([3, BIG], [1], 1, 10, 0, closure=(2, 3, 1)), list(e))

# ---- T7: M/M/1 (rho=0.5) y M/M/1/1 contra teoria, 30 replicas ----
print("\n[T7] M/M/1 (lam=1, mu=2): teoria Lq=0.5, Wq=0.5, L=1, P_wait=0.5")
r = new.run_replicas(30, 7, lam=1.0, mu=2.0, c=1, horizon=20000, warmup=500)
s7 = new.summarize(r)
for k, v in (("Lq", 0.5), ("Wq", 0.5), ("L", 1.0), ("P_wait", 0.5)):
    m, lo, hi = s7[k]
    print(f"   {k}: {m:.4f} [{lo:.4f},{hi:.4f}] teoria {v} -> {'dentro' if lo <= v <= hi else 'FUERA'}")
    results.append({"prueba": f"T7 M/M/1 {k}", "teoria": v, "media": m, "lo": lo, "hi": hi})
r = new.run_replicas(30, 8, lam=1.0, mu=1.0, c=1, horizon=20000, warmup=500, wait=False)
m, lo, hi = new.ci95([x["P_block"] for x in r])
print(f"[T7b] M/M/1/1 (a=1): P_block {m:.4f} [{lo:.4f},{hi:.4f}] teoria 0.5")
results.append({"prueba": "T7b M/M/1/1 P_block", "teoria": 0.5, "media": m, "lo": lo, "hi": hi})

# ---- T8: cobertura del IC con pocas replicas (original fija t=2.045 si n=30, 1.96 en otro caso) ----
Lt = th.mmcc(3.2, 0.25, 17)["L"]
print(f"\n[T8] cobertura del IC95% para L (M/M/17/17, L teorica={Lt:.4f}) con n_rep=5, 300 lotes")
cov_o = cov_n = 0
NB = 300
for b in range(NB):
    o = orig.replicate(3.2, 0.25, 17, 400, 50, n_rep=5, seed=1000 * b, wait=False)["L"]
    cov_o += o[1] <= Lt <= o[2]
    rr = new.run_replicas(5, 5000 + b, lam=3.2, mu=0.25, c=17, horizon=400, warmup=50, wait=False)
    n = new.ci95([x["L"] for x in rr])
    cov_n += n[1] <= Lt <= n[2]
print(f"   original : {cov_o / NB:.3f}   corregido: {cov_n / NB:.3f}   (nominal 0.95; EE ~ 0.013)")
results.append({"prueba": "T8 cobertura IC n_rep=5", "original": cov_o / NB, "corregido": cov_n / NB, "lotes": NB})

# ---- T9: solapamiento de semillas del original ----
a = [orig.simulate(3.2, 0.25, 17, 300, 50, 1 + i, False)["L"] for i in range(30)]
b = [orig.simulate(3.2, 0.25, 17, 300, 50, 2 + i, False)["L"] for i in range(30)]
shared = sum(1 for x in a for y in b if x == y)
print(f"\n[T9] original: corridas con semilla base 1 y base 2 comparten {shared} de 30 replicas identicas")
results.append({"prueba": "T9 solapamiento semillas", "compartidas": shared})
Ls = np.array([orig.simulate(3.2, 0.25, 17, 300, 50, 10 + i, False)["L"] for i in range(400)])
rho1 = float(np.corrcoef(Ls[:-1], Ls[1:])[0, 1])
print(f"     autocorrelacion lag-1 de L entre semillas consecutivas (400 replicas): {rho1:.3f} (EE~0.05)")
results.append({"prueba": "T9b autocorr semillas", "rho1": rho1})

# ---- T10: sesgo por fin de horizonte en un caso real (mismas semillas, vaciado vs censura) ----
print("\n[T10] sesgo de censura al fin del horizonte (M/M/17, H=100, W=20, 400 replicas)")
kw = dict(lam=3.2, mu=0.25, c=17, horizon=100, warmup=20)
d = [new.simulate(rng=r, drain=True, **kw) for r in new.make_rngs(99, 400)]
nd = [new.simulate(rng=r, drain=False, **kw) for r in new.make_rngs(99, 400)]
mm = th.mmc(3.2, 0.25, 17)
for k in ("Wq", "L", "Lq", "P_wait"):
    md, mn = np.mean([x[k] for x in d]), np.mean([x[k] for x in nd])
    tk = {"P_wait": mm["C"]}.get(k, mm.get(k))
    print(f"   {k}: con vaciado {md:.4f}  censurado {mn:.4f}  dif rel {(mn - md) / md * 100:+.2f}%  (teoria {tk:.4f})")
    results.append({"prueba": f"T10 censura {k}", "drain": md, "censurado": mn})

# ---- T11: error al usar lambda en vez de lambda_ef (M/M/c/c) ----
sim = new.run_replicas(30, 11, lam=3.2, mu=0.25, c=17, horizon=2000, warmup=200, wait=False)
L = np.mean([x["L"] for x in sim])
W = np.mean([x["W"] for x in sim])
le = np.mean([x["lam_ef"] for x in sim])
print(f"\n[T11] M/M/c/c: L={L:.4f}, lam_ef*W={le * W:.4f} (err {abs(L - le * W) / L * 100:.2f}%), "
      f"lam*W={3.2 * W:.4f} (err {abs(L - 3.2 * W) / L * 100:.2f}%)")
results.append({"prueba": "T11 Little lam vs lam_ef", "L": L, "lam_ef_W": le * W, "lam_W": 3.2 * W})

json.dump(results, open("resultados/pruebas.json", "w"), indent=1, default=float)
