"""Simulador de eventos discretos de una UCI: M/M/c (con espera) y M/M/c/c (sin espera).

Entradas: lam, mu, c, horizonte, calentamiento (warm-up) y semilla.
Salidas: L, Lq, Wq, utilizacion, prob. de esperar, prob. de bloqueo.
"""
import heapq
import math
import random
import statistics

ARRIVAL = 0
DEPARTURE = 1


def simulate(lam, mu, c, horizon, warmup, seed, wait=True):
    rng = random.Random(seed)
    events = []  # heap de (tiempo, tipo)
    heapq.heappush(events, (rng.expovariate(lam), ARRIVAL))

    t = 0.0
    last_t = 0.0
    busy = 0            # camas ocupadas
    queue = []          # tiempos de llegada de los pacientes en espera (FIFO)

    # acumuladores (solo despues del calentamiento)
    area_L = 0.0
    area_Lq = 0.0
    area_busy = 0.0
    n_arrivals = 0
    n_waited = 0
    n_blocked = 0
    waits = []

    while events:
        t, kind = heapq.heappop(events)
        if t > horizon:
            break

        # integrar las areas desde el ultimo evento hasta ahora
        if t > warmup:
            dt = t - max(last_t, warmup)
            area_L += (busy + len(queue)) * dt
            area_Lq += len(queue) * dt
            area_busy += busy * dt
        last_t = t

        if kind == ARRIVAL:
            heapq.heappush(events, (t + rng.expovariate(lam), ARRIVAL))
            counted = t > warmup
            if counted:
                n_arrivals += 1
            if busy < c:
                busy += 1
                heapq.heappush(events, (t + rng.expovariate(mu), DEPARTURE))
                if counted:
                    waits.append(0.0)
            elif wait:
                queue.append(t)
                if counted:
                    n_waited += 1
            else:
                if counted:
                    n_blocked += 1
        else:  # DEPARTURE
            if queue:
                arrival_time = queue.pop(0)
                heapq.heappush(events, (t + rng.expovariate(mu), DEPARTURE))
                if arrival_time > warmup:
                    waits.append(t - arrival_time)
            else:
                busy -= 1

    obs = horizon - warmup
    return {
        "L": area_L / obs,
        "Lq": area_Lq / obs,
        "Wq": statistics.mean(waits) if waits else 0.0,
        "util": area_busy / obs / c,
        "P_wait": n_waited / n_arrivals if n_arrivals else 0.0,
        "P_block": n_blocked / n_arrivals if n_arrivals else 0.0,
    }


def replicate(lam, mu, c, horizon, warmup, n_rep=30, seed=12345, wait=True):
    """Corre n_rep replicas y devuelve media e IC 95% de cada medida."""
    runs = [simulate(lam, mu, c, horizon, warmup, seed + i, wait) for i in range(n_rep)]
    out = {}
    t_crit = 2.045 if n_rep == 30 else 1.96  # t de Student para 29 g.l.
    for key in runs[0]:
        vals = [r[key] for r in runs]
        m = statistics.mean(vals)
        s = statistics.stdev(vals)
        h = t_crit * s / math.sqrt(n_rep)
        out[key] = (m, m - h, m + h)
    return out


# ---- teoria para comparar ----
def erlang_b(c, a):
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    b = erlang_b(c, a)
    rho = a / c
    return b / (1 - rho * (1 - b))


if __name__ == "__main__":
    lam, mu, c = 3.2, 0.25, 17
    a = lam / mu
    C = erlang_c(c, a)
    Lq = C * (a / c) / (1 - a / c)
    print("=== M/M/c con espera ===")
    for k, (m, lo, hi) in replicate(lam, mu, c, 2000, 200, 30, wait=True).items():
        print(f"{k:8s} {m:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
    print(f"Teoria: Lq={Lq:.4f} Wq={Lq / lam:.4f} P_wait={C:.4f} L={Lq + a:.4f}")
    print("=== M/M/c/c sin espera ===")
    for k, (m, lo, hi) in replicate(lam, mu, c, 2000, 200, 30, wait=False).items():
        print(f"{k:8s} {m:.4f}  IC95% [{lo:.4f}, {hi:.4f}]")
    B = erlang_b(c, a)
    print(f"Teoria: B={B:.4f} L={a * (1 - B):.4f}")
