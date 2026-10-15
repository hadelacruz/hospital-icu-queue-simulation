"""Teoria de colas M/M/c y M/M/c/c para el Hospital Regional del Altiplano (grupo 7).

Formulas (diapositivas de la sesion de teoria de colas):
  a = lambda/mu                       carga ofrecida (Erlangs)
  rho = a/c                           utilizacion por cama
  B(c,a): recurrencia de Erlang B     B(0)=1, B(k) = a B(k-1) / (k + a B(k-1))
  C(c,a) = B / (1 - rho (1 - B))      Erlang C (prob. de esperar, PASTA)
  Lq = C rho/(1-rho),  Wq = Lq/lambda,  W = Wq + 1/mu,  L = Lq + a
"""
import math

G = 7
MU = 0.25
LAM = 3.0 + 0.2 * ((G - 1) % 5)
C = 17 + ((G - 1) % 3)
A = LAM / MU
RHO = A / C
K_OFF = math.ceil(C - A) + 3
T_B = 7.0


def erlang_b(c, a):
    """Recurrencia estable de Erlang B."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Probabilidad de esperar (Erlang C); requiere a < c."""
    b = erlang_b(c, a)
    rho = a / c
    return b / (1.0 - rho * (1.0 - b))


def mmc(lam, mu, c):
    """Medidas estacionarias M/M/c (tiempos en dias)."""
    a = lam / mu
    rho = a / c
    cc = erlang_c(c, a)
    lq = cc * rho / (1 - rho)
    wq = lq / lam
    w = wq + 1 / mu
    return dict(a=a, rho=rho, C=cc, Lq=lq, Wq=wq, W=w, L=lq + a)


def mmcc(lam, mu, c):
    """Medidas M/M/c/c (Erlang B / perdida)."""
    a = lam / mu
    b = erlang_b(c, a)
    lam_ef = lam * (1 - b)
    L = lam_ef / mu
    return dict(a=a, B=b, lam_ef=lam_ef, L=L, util=L / c)


def min_c_b(lam, mu, meta):
    a = lam / mu
    c = 1
    while erlang_b(c, a) > meta:
        c += 1
    return c


if __name__ == "__main__":
    print(f"g={G} lambda={LAM} c={C} a={A} rho={RHO:.6f} k_off={K_OFF} c-k_off={C-K_OFF}")
    m = mmc(LAM, MU, C)
    for k, v in m.items():
        print(f"  M/M/c {k} = {v:.6f}")
    print(f"  Wq = {m['Wq']*24:.4f} h   W = {m['W']*24:.4f} h")
    s = mmcc(LAM, MU, C)
    for k, v in s.items():
        print(f"  M/M/c/c {k} = {v:.6f}")
    # Little
    print("Little M/M/c: Lq - lam*Wq =", m['Lq'] - LAM * m['Wq'], " L - lam*W =", m['L'] - LAM * m['W'])
    print("Little M/M/c/c: L - lam_ef/mu =", s['L'] - s['lam_ef'] * (1 / MU))
    print("Error si se usa lambda: L' = lam*(1/mu) =", LAM / MU, "vs L =", s['L'])
    # numeros para 2.1
    print("--- numeros para predicciones 2.1 ---")
    for cs2, nm in [(0, 'Cs=0'), (1, 'Cs=1'), (4, 'Cs=2')]:
        print(nm, "Wq AC (h) =", m['Wq'] * 24 * (1 + cs2) / 2)
    gr = LAM - (C - K_OFF) * MU
    dr = C * MU - LAM
    print("tasa acumulacion durante cierre =", gr, " tasa drenaje =", dr)
    print("Td formula =", gr * T_B / dr, "dias;  Q_pico =", gr * T_B)
    b0 = erlang_b(C, A)
    mu2 = MU / 1.25
    b1 = erlang_b(C, LAM / mu2)
    print(f"B actual = {b0:.6f}; B con estancia +25% (a={LAM/mu2:.2f}) = {b1:.6f}; razon = {b1/b0:.4f}")
