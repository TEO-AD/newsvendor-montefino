"""
=======================================================================
  MODELO NEWSVENDOR — Montefino S.A.C. (Yogurt Artesanal, Ayacucho)
  Versión final — todos los parámetros son datos reales de la empresa
=======================================================================

PARÁMETROS Y FUENTES:
─────────────────────────────────────────────────────────────────────
  Precio de lista (c/ IGV)  : S/ 13.50 / litro  → dato empresa
  IGV Perú                  : 18 %               → D.S. 055-99-EF
  Precio neto (sin IGV)  p  : 13.50 / 1.18
                            = S/ 11.44 / litro   → precio real Montefino
  Costo de producción    c  : S/ 8.00  / litro   → dato empresa
  Valor de rescate       s  : S/ 0.00             → dato empresa
                              (lote deteriorado sin valor recuperable)

  Cu = p − c = 3.44   Costo de subproducción (utilidad perdida)
  Co = c − s = 8.00   Costo de sobreproducción (merma total del lote)

  Ratio crítico CR = Cu / (Cu + Co) = 3.44 / 11.44 ≈ 0.3007

DISTRIBUCIÓN DE LA DEMANDA SEMANAL:
  μ = 400 litros / semana   → producción semanal Montefino
    Ayacucho (80 %) = 320 L     Lima (20 %) = 80 L
  σ = 47  litros / semana   → variabilidad reportada por la empresa
  Demanda ~ Normal(400, 47)
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ── PARÁMETROS ────────────────────────────────────────────────────────
p_lista = 13.50
igv     = 0.18
p       = p_lista / (1 + igv)

c = 8.00
s = 0.00

Cu = p - c
Co = c - s
CR = Cu / (Cu + Co)

mu    = 400.0
sigma = 47.0

Q_star = norm.ppf(CR, loc=mu, scale=sigma)

# ── RESULTADOS EN CONSOLA ─────────────────────────────────────────────
print("=" * 60)
print("  RESULTADOS — NEWSVENDOR MONTEFINO S.A.C.")
print("=" * 60)
print(f"  Precio lista  (c/IGV)       : S/ {p_lista:.2f}")
print(f"  Precio neto   (sin IGV)  p  : S/ {p:.4f}")
print(f"  Costo producción         c  : S/ {c:.2f}")
print(f"  Valor de rescate         s  : S/ {s:.2f}")
print(f"  Cu = p − c                  : S/ {Cu:.4f}")
print(f"  Co = c − s                  : S/ {Co:.2f}")
print("-" * 60)
print(f"  Ratio crítico  CR           : {CR:.4f}  ({CR*100:.2f} %)")
print(f"  Demanda ~ N(μ={mu:.0f}, σ={sigma:.0f})  litros/semana")
print(f"  Cantidad óptima  Q*         : {Q_star:.2f} litros/semana")
print(f"  Ahorro vs producción actual : {mu - Q_star:.2f} litros/semana")
print("=" * 60)

# ── VECTORES ──────────────────────────────────────────────────────────
Q_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 600)
F_Q   = norm.cdf(Q_range, mu, sigma)
pdf_Q = norm.pdf(Q_range, mu, sigma)

def E_cost(Q):
    over  = Co * ((Q-mu)*norm.cdf(Q,mu,sigma) + sigma*norm.pdf(Q,mu,sigma))
    under = Cu * ((mu-Q)*(1-norm.cdf(Q,mu,sigma)) + sigma*norm.pdf(Q,mu,sigma))
    return over + under

def E_profit(Q):
    E_sales = mu - (mu-Q)*norm.cdf(Q,mu,sigma) - sigma*norm.pdf(Q,mu,sigma)
    return p*E_sales - c*Q

EC  = np.array([E_cost(q)   for q in Q_range])
EP  = np.array([E_profit(q) for q in Q_range])

EC_over  = np.array([
    Co * ((q-mu)*norm.cdf(q,mu,sigma) + sigma*norm.pdf(q,mu,sigma))
    for q in Q_range
])

EC_under = np.array([
    Cu * ((mu-q)*(1-norm.cdf(q,mu,sigma)) + sigma*norm.pdf(q,mu,sigma))
    for q in Q_range
])

# ── COLORES ───────────────────────────────────────────────────────────
BLUE   = "#2A6DB5"
GREEN  = "#1DA57A"
RED    = "#D94F3D"
ORANGE = "#E8821A"
DARK   = "#1C2331"
GRAY   = "#8C95A0"
GOLD   = "#C9A84C"
LIGHT  = "#F7F9FC"

# ── CARPETA DE SALIDA ─────────────────────────────────────────────────
output_dir = "graficas_montefino"
os.makedirs(output_dir, exist_ok=True)

# ── FUNCIÓN PARA DAR ESTILO A CADA GRÁFICA ────────────────────────────
def preparar_grafica(ax, titulo):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors=DARK, labelsize=11)
    ax.set_title(titulo, fontsize=15, fontweight="bold", color=DARK, pad=15)
    ax.grid(alpha=0.20)

def guardar_grafica(fig, nombre_archivo):
    ruta = os.path.join(output_dir, nombre_archivo)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor=LIGHT)
    print(f"✔ Gráfica guardada: {ruta}")
    plt.show()
    plt.close(fig)

# =====================================================================
# GRÁFICA 1: CDF + RATIO CRÍTICO
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
fig.patch.set_facecolor(LIGHT)

preparar_grafica(ax1, "1. CDF de Demanda y Ratio Crítico")

ax1.plot(Q_range, F_Q, color=BLUE, lw=2.8, label="F(Q) — CDF demanda")
ax1.axhline(
    CR, color=GOLD, lw=2, ls="--",
    label=f"Ratio crítico CR = {CR:.3f} ({CR*100:.1f} %)"
)
ax1.axvline(
    Q_star, color=RED, lw=2.2, ls=":",
    label=f"Q* = {Q_star:.1f} L/sem"
)
ax1.fill_betweenx([0, CR], Q_range[0], Q_star, alpha=0.08, color=BLUE)

ax1.annotate(
    f"Q* = {Q_star:.1f} L",
    xy=(Q_star, CR),
    xytext=(Q_star + 10, CR - 0.13),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    fontsize=11,
    color=RED,
    fontweight="bold"
)

ax1.set_xlabel("Cantidad Q (litros/semana)", color=DARK, fontsize=12)
ax1.set_ylabel("Probabilidad acumulada F(Q)", color=DARK, fontsize=12)
ax1.set_ylim(0, 1.05)
ax1.set_xlim(Q_range[0], Q_range[-1])
ax1.legend(fontsize=10, framealpha=0.9)

guardar_grafica(fig, "01_cdf_ratio_critico.png")

# =====================================================================
# GRÁFICA 2: PDF CON ÁREAS DE RIESGO
# =====================================================================
fig, ax2 = plt.subplots(figsize=(10, 6), dpi=120)
fig.patch.set_facecolor(LIGHT)

preparar_grafica(ax2, "2. Distribución de Demanda Semanal")

ax2.plot(
    Q_range, pdf_Q, color=GREEN, lw=2.8,
    label=f"Demanda ~ N(μ={mu:.0f}, σ={sigma:.0f})"
)
ax2.axvline(Q_star, color=RED, lw=2.2, ls=":", label=f"Q* = {Q_star:.1f} L")
ax2.axvline(mu, color=GRAY, lw=1.8, ls="--", label=f"μ = {mu:.0f} L")

ax2.fill_between(
    Q_range, pdf_Q, where=(Q_range <= Q_star),
    alpha=0.25, color=RED,
    label=f"P(D ≤ Q*): sobrante posible {CR*100:.1f} %"
)

ax2.fill_between(
    Q_range, pdf_Q, where=(Q_range > Q_star),
    alpha=0.18, color=GREEN,
    label=f"P(D > Q*): riesgo de escasez {(1-CR)*100:.1f} %"
)

ax2.set_xlabel("Demanda (litros/semana)", color=DARK, fontsize=12)
ax2.set_ylabel("Densidad de probabilidad", color=DARK, fontsize=12)
ax2.set_xlim(Q_range[0], Q_range[-1])
ax2.legend(fontsize=10, framealpha=0.9)

guardar_grafica(fig, "02_distribucion_demanda.png")

# =====================================================================
# GRÁFICA 3: COSTO ESPERADO
# =====================================================================
fig, ax3 = plt.subplots(figsize=(10, 6), dpi=120)
fig.patch.set_facecolor(LIGHT)

preparar_grafica(ax3, "3. Costo Esperado vs. Cantidad Producida")

ax3.plot(Q_range, EC, color=DARK, lw=2.8, label="Costo esperado total")
ax3.plot(
    Q_range, EC_over, color=ORANGE, lw=2, ls="--",
    label=f"Sobreproducción (Co = S/{Co:.2f})"
)
ax3.plot(
    Q_range, EC_under, color=BLUE, lw=2, ls="--",
    label=f"Subproducción (Cu = S/{Cu:.2f})"
)

ax3.axvline(Q_star, color=RED, lw=2.2, ls=":", label=f"Q* = {Q_star:.1f} L")

min_ec = E_cost(Q_star)
ax3.scatter([Q_star], [min_ec], color=RED, zorder=5, s=90)

ax3.annotate(
    f"Mínimo = S/ {min_ec:.2f}",
    xy=(Q_star, min_ec),
    xytext=(Q_star + 10, min_ec + 2),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    fontsize=11,
    color=RED,
    fontweight="bold"
)

ax3.set_xlabel("Cantidad Q (litros/semana)", color=DARK, fontsize=12)
ax3.set_ylabel("Costo esperado (S/)", color=DARK, fontsize=12)
ax3.set_xlim(Q_range[0], Q_range[-1])
ax3.legend(fontsize=10, framealpha=0.9)

guardar_grafica(fig, "03_costo_esperado.png")

# =====================================================================
# GRÁFICA 4: BENEFICIO ESPERADO
# =====================================================================
fig, ax4 = plt.subplots(figsize=(10, 6), dpi=120)
fig.patch.set_facecolor(LIGHT)

preparar_grafica(ax4, "4. Beneficio Esperado vs. Cantidad Producida")

ax4.plot(Q_range, EP, color=GREEN, lw=2.8, label="Beneficio esperado")
ax4.axvline(Q_star, color=RED, lw=2.2, ls=":", label=f"Q* = {Q_star:.1f} L")
ax4.axvline(mu, color=GRAY, lw=1.8, ls="--", label=f"Producción actual μ = {mu:.0f} L")
ax4.axhline(0, color=GRAY, lw=1)

max_ep = E_profit(Q_star)
ep_mu  = E_profit(mu)

ax4.scatter([Q_star], [max_ep], color=RED, zorder=5, s=90)
ax4.scatter([mu], [ep_mu], color=GRAY, zorder=5, s=90)

ax4.fill_between(Q_range, EP, 0, where=(EP >= 0), alpha=0.12, color=GREEN)
ax4.fill_between(Q_range, EP, 0, where=(EP < 0), alpha=0.12, color=RED)

ax4.annotate(
    f"Máximo = S/ {max_ep:.2f}\nQ* = {Q_star:.1f} L",
    xy=(Q_star, max_ep),
    xytext=(Q_star - 90, max_ep - 18),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    fontsize=11,
    color=RED,
    fontweight="bold"
)

ax4.annotate(
    f"Actual = S/ {ep_mu:.2f}\nμ = {mu:.0f} L",
    xy=(mu, ep_mu),
    xytext=(mu + 10, ep_mu - 18),
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5),
    fontsize=11,
    color=DARK
)

ax4.set_xlabel("Cantidad Q (litros/semana)", color=DARK, fontsize=12)
ax4.set_ylabel("Beneficio esperado (S/)", color=DARK, fontsize=12)
ax4.set_xlim(Q_range[0], Q_range[-1])
ax4.legend(fontsize=10, framealpha=0.9)

guardar_grafica(fig, "04_beneficio_esperado.png")

print("\n Listo. Se generaron las 4 gráficas por separado en alta calidad.")
print(f" Carpeta: {output_dir}")