"""Task 1.4a: promedio movil (metodo de Welch) para justificar calentamiento y horizonte.
Se simulan R replicas ARRANCANDO VACIO, se promedia entre replicas la longitud de fila Q(t) y el
numero en el sistema N(t) en una malla diaria, y se suaviza con una ventana movil."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import icu_sim as s

R, H, STEP, WIN = 300, 300, 1.0, 10
grid = np.arange(0.0, H, STEP)
out = {}
for wait, key in ((True, "Q"), (False, "N")):
    runs = [s.simulate(3.2, 0.25, 17, H, 0, r, wait=wait, grid=grid) for r in s.make_rngs(2024, R)]
    arr = np.array([x["grid_Q"] if key == "Q" else x["grid_N"] for x in runs])
    out[key] = arr.mean(axis=0)


def movavg(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
a = ax[0]
a.plot(grid, out["Q"], color="#bbbbbb", lw=0.8, label="promedio entre 300 réplicas")
a.plot(grid[WIN - 1:], movavg(out["Q"], WIN), color="#1f5fa8", lw=1.8, label=f"promedio móvil (ventana {WIN} d)")
a.axhline(0.6013, color="k", ls="--", lw=1, label=r"$L_q$ teórico (0.601)")
a.axvline(100, color="#c0392b", ls=":", lw=1.5, label="calentamiento = 100 d")
a.set_xlabel("tiempo (días)"); a.set_ylabel("longitud de la fila"); a.set_title("Variante con espera (M/M/17)")
a.legend(fontsize=7)
a = ax[1]
a.plot(grid, out["N"], color="#bbbbbb", lw=0.8)
a.plot(grid[WIN - 1:], movavg(out["N"], WIN), color="#1f5fa8", lw=1.8)
a.axhline(12.0672, color="k", ls="--", lw=1)
a.axvline(100, color="#c0392b", ls=":", lw=1.5)
a.set_xlabel("tiempo (días)"); a.set_ylabel("pacientes en el sistema"); a.set_title("Variante sin espera (M/M/17/17)")
plt.tight_layout()
plt.savefig("figuras/welch.pdf"); plt.savefig("figuras/welch.png", dpi=150)

# ventana movil de 10 d: valor del promedio movil en distintos instantes vs referencia (t>=150)
for key in ("Q", "N"):
    ref = out[key][150:].mean()
    ma = movavg(out[key], WIN)
    print(key, "referencia estacionaria (t>=150) =", round(ref, 4))
    for t in (10, 20, 30, 40, 50, 60, 80, 100, 150):
        v = ma[t - WIN + 1] if t >= WIN else float("nan")
        print(f"   t={t:4d} (ventana [{t-WIN+1},{t}]): {v:.4f}  ({(v-ref)/ref*100:+.1f}%)")
    dev = np.abs(ma - ref) / ref
    for tol in (0.02, 0.05, 0.10):
        ok = dev <= tol
        idx = next((i for i in range(len(ok)) if ok[i:].all()), None)
        print(f"   queda dentro de +-{int(tol*100)}% desde t =", None if idx is None else grid[idx + WIN - 1])
