"""Simulador de eventos discretos CORREGIDO de la UCI (M/M/c, M/M/c/c, estancia general, cierre parcial).

Relacion con las formulas de las diapositivas
  * Llegadas Poisson(lam): tiempos entre llegadas exponenciales de media 1/lam.
  * Estancia: exponencial (Cs=1), determinista (Cs=0) o lognormal con
        s^2 = ln(1+Cs^2),  m = ln(1/mu) - s^2/2   (media 1/mu para todas).
  * L, Lq, utilizacion = promedios EN EL TIEMPO (integral de la trayectoria / tiempo observado):
        L = (1/T) int N(t) dt,  Lq = (1/T) int Q(t) dt,  rho_hat = (1/(cT)) int B(t) dt.
  * P_wait (= C(c,a) en teoria, por PASTA) y P_block (= B(c,a)) son fracciones de LLEGADAS.
  * Wq, W: promedio sobre pacientes, medidos desde la LLEGADA (Wq = inicio de servicio - llegada).
  * lam_ef = pacientes admitidos / T  (en M/M/c/c es lam (1 - B)).  Ley de Little: L = lam_ef W,
    Lq = lam_ef Wq.
  * Calentamiento: las integrales cubren solo (warmup, horizonte] y solo se cuentan los pacientes que
    LLEGAN en (warmup, horizonte].
  * Fin de horizonte (drain=True): se dejan de generar llegadas en t = horizonte pero se sigue
    simulando hasta vaciar la lista de eventos, de modo que TODO paciente contado tiene su espera y su
    estadia completas (sin censura). Las integrales se recortan en t = horizonte.
  * Cierre parcial: closure=(t0, Tb, k_off). La capacidad es c - k_off en [t0, t0+Tb). Un paciente que
    ya esta en cama NO se expulsa; las camas "cerradas" son las que se van liberando.
  * Replicas independientes: numpy SeedSequence.spawn -> flujos PCG64 estadisticamente independientes.
"""
import heapq
import math
from collections import deque

import numpy as np
from scipy import stats

ARR, DEP, CLOSE, OPEN = 0, 1, 2, 3


def make_service_sampler(mu, dist="exp", cs=1.0):
    mean = 1.0 / mu
    if dist == "exp":
        return lambda rng: rng.exponential(mean)
    if dist == "det":
        return lambda rng: mean
    if dist == "lognormal":
        s2 = math.log(1.0 + cs * cs)
        s = math.sqrt(s2)
        m = math.log(mean) - s2 / 2.0
        return lambda rng: rng.lognormal(m, s)
    raise ValueError(dist)


def simulate(lam, mu, c, horizon, warmup, rng, wait=True, service="exp", cs=1.0,
             closure=None, grid=None, drain=True, interarrival=None, service_fn=None):
    """Una replica. rng: numpy.random.Generator. interarrival/service_fn: ganchos para pruebas."""
    ia = interarrival or (lambda r: r.exponential(1.0 / lam))
    sv = service_fn or make_service_sampler(mu, service, cs)

    ev = []
    seq = 0

    def push(time, kind):
        nonlocal seq
        heapq.heappush(ev, (time, seq, kind))
        seq += 1

    t_first = ia(rng)
    if t_first <= horizon:
        push(t_first, ARR)
    if closure is not None:
        t0, tb, k_off = closure
        push(t0, CLOSE)
        push(t0 + tb, OPEN)

    cap = c
    busy = 0
    queue = deque()  # (tiempo de llegada, contado?)
    last_t = 0.0
    area_n = area_q = area_b = 0.0
    n_arr = n_wait = n_block = n_adm = 0
    waits, sojourns = [], []
    gi = 0
    gq = gn = None
    if grid is not None:
        gq = np.zeros(len(grid))
        gn = np.zeros(len(grid))

    def advance(t_new):
        nonlocal last_t, area_n, area_q, area_b, gi
        lo = max(last_t, warmup)
        hi = min(t_new, horizon)
        if hi > lo:
            dt = hi - lo
            area_n += (busy + len(queue)) * dt
            area_q += len(queue) * dt
            area_b += busy * dt
        if grid is not None:
            while gi < len(grid) and grid[gi] < t_new:
                gq[gi] = len(queue)
                gn[gi] = busy + len(queue)
                gi += 1
        last_t = t_new

    def start_service(t, arrival_time, counted):
        nonlocal busy, n_adm
        s = sv(rng)
        busy += 1
        push(t + s, DEP)
        if counted:
            n_adm += 1
            waits.append(t - arrival_time)
            sojourns.append(t - arrival_time + s)

    def fill(t):
        while queue and busy < cap:
            a_t, counted = queue.popleft()
            start_service(t, a_t, counted)

    while ev:
        t, _, kind = heapq.heappop(ev)
        if not drain and t > horizon:
            break
        advance(t)
        if kind == ARR:
            tn = t + ia(rng)
            if tn <= horizon:
                push(tn, ARR)
            counted = warmup < t <= horizon
            if counted:
                n_arr += 1
            if busy < cap:
                start_service(t, t, counted)
            elif wait:
                queue.append((t, counted))
                if counted:
                    n_wait += 1
            elif counted:
                n_block += 1
        elif kind == DEP:
            busy -= 1
            fill(t)
        elif kind == CLOSE:
            cap = c - closure[2]
        else:  # OPEN
            cap = c
            fill(t)
    advance(max(horizon, last_t))

    obs = horizon - warmup
    out = {
        "L": area_n / obs,
        "Lq": area_q / obs,
        "util": area_b / obs / c,
        "P_wait": n_wait / n_arr if n_arr else float("nan"),
        "P_block": n_block / n_arr if n_arr else float("nan"),
        "Wq": float(np.mean(waits)) if waits else float("nan"),
        "W": float(np.mean(sojourns)) if sojourns else float("nan"),
        "lam_ef": n_adm / obs,
        "n_arr": n_arr,
        "n_block": n_block,
        "n_adm": n_adm,
    }
    if grid is not None:
        out["grid_Q"] = gq
        out["grid_N"] = gn
    return out


def make_rngs(seed, n):
    """n flujos independientes (SeedSequence.spawn)."""
    return [np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(n)]


def run_replicas(n_rep, seed, **kw):
    return [simulate(rng=r, **kw) for r in make_rngs(seed, n_rep)]


def ci95(values):
    """Media e IC 95% t-Student con la REPLICA como unidad de muestreo."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = v.mean()
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def summarize(runs, keys=("L", "Lq", "Wq", "W", "util", "P_wait", "P_block", "lam_ef")):
    return {k: ci95([r[k] for r in runs]) for k in keys}
