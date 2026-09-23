"""
Genera las figuras (elaboración propia) del Capítulo 2 - Marco Referencial.
Uso:  python scripts/figuras_marco.py
Salida: imagenes/02-marco-referencial/*.pdf
Requiere: numpy, scipy, matplotlib
Las señales son sintéticas (ilustrativas); no provienen del dataset.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUT = os.path.join(os.path.dirname(__file__), "..", "imagenes", "02-marco-referencial")
os.makedirs(OUT, exist_ok=True)

GUINDA = "#7A142A"
AZUL = "#1F4E79"
GRIS = "#6B6B6B"
GRIS_CLARO = "#BDBDBD"
ORO = "#B8860B"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

rng = np.random.default_rng(7)


def guardar(fig, nombre):
    fig.savefig(os.path.join(OUT, nombre))
    plt.close(fig)


def ruido_rosa(n, fs, escala=1.0):
    """Ruido tipo 1/f para simular EEG de fondo."""
    blanco = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / fs)
    espectro = np.fft.rfft(blanco)
    f[0] = f[1]
    espectro /= np.sqrt(f)
    x = np.fft.irfft(espectro, n)
    return escala * x / np.std(x)


# ---------------------------------------------------------------------
# 1. Ritmos cerebrales
# ---------------------------------------------------------------------
def fig_ritmos():
    fs = 250
    t = np.arange(0, 2, 1 / fs)
    bandas = [
        (r"Delta $\delta$ (0.5–4 Hz)", 2.0, 1.00),
        (r"Theta $\theta$ (4–8 Hz)", 6.0, 0.70),
        (r"Alfa $\alpha$ (8–13 Hz)", 10.0, 0.60),
        (r"Beta $\beta$ (13–30 Hz)", 20.0, 0.30),
        (r"Gamma $\gamma$ (>30 Hz)", 40.0, 0.18),
    ]
    fig, axs = plt.subplots(len(bandas), 1, figsize=(6.2, 4.6), sharex=True)
    for ax, (nombre, f0, a) in zip(axs, bandas):
        # oscilación con amplitud modulada + un poco de ruido
        env = 1 + 0.3 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
        x = a * env * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6))
        x += 0.05 * rng.standard_normal(t.size)
        ax.plot(t, x, color=AZUL, lw=0.9)
        ax.set_ylim(-1.6, 1.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(-0.02, 0.5, nombre, transform=ax.transAxes, ha="right", va="center")
    axs[-1].set_xlabel("Tiempo (s)")
    fig.subplots_adjust(left=0.30, hspace=0.25)
    guardar(fig, "ritmos_eeg.pdf")


# ---------------------------------------------------------------------
# 2. ERP: forma de onda y efecto del promediado
# ---------------------------------------------------------------------
def erp_plantilla(t, objetivo=True):
    """Onda ERP sintética (µV) con N100, P200, N200 y P300 (si es objetivo)."""
    g = lambda mu, s, a: a * np.exp(-0.5 * ((t - mu) / s) ** 2)
    x = g(0.10, 0.02, -3.0) + g(0.18, 0.03, 2.5) + g(0.24, 0.025, -2.0)
    if objetivo:
        x += g(0.33, 0.06, 7.5)
    return x


def fig_erp():
    fs = 240
    t = np.arange(-0.1, 0.8, 1 / fs)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))

    ax = axs[0]
    ax.plot(t * 1000, erp_plantilla(t, True), color=GUINDA, lw=1.6, label="Estímulo objetivo (raro)")
    ax.plot(t * 1000, erp_plantilla(t, False), color=GRIS, lw=1.3, ls="--", label="Estímulo no objetivo")
    for nom, tt, xy in [("N100", 100, (40, -4.4)), ("P200", 180, (150, 5.0)),
                        ("N200", 240, (275, -4.4)), ("P300", 330, (200, 9.3))]:
        y = erp_plantilla(np.array([tt / 1000]), True)[0]
        ax.annotate(nom, (tt, y), xy, ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.6))
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("Tiempo desde el estímulo (ms)")
    ax.set_ylabel(r"Amplitud ($\mu$V)")
    ax.set_title("(a) Componentes del ERP")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_ylim(-5.5, 12)

    ax = axs[1]
    for n, col, lw in [(1, GRIS_CLARO, 0.7), (10, GRIS, 0.9), (60, GUINDA, 1.6)]:
        bb, aa = signal.butter(4, [0.5, 20], btype="bandpass", fs=fs)
        def ruido():
            r = signal.filtfilt(bb, aa, rng.standard_normal(t.size))
            return 8 * r / np.std(r)
        ensayos = np.array([erp_plantilla(t) + ruido() for _ in range(n)])
        ax.plot(t * 1000, ensayos.mean(0), color=col, lw=lw, label=f"N = {n}")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("Tiempo desde el estímulo (ms)")
    ax.set_title("(b) Promediado de N ensayos")
    ax.set_ylim(-14, 18)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    guardar(fig, "erp_componentes_promediado.pdf")


# ---------------------------------------------------------------------
# 3. Paradigma oddball: secuencia de estímulos
# ---------------------------------------------------------------------
def fig_oddball():
    n = 24
    raros = rng.random(n) < 0.18
    raros[[5, 13, 20]] = True
    fig, ax = plt.subplots(figsize=(6.3, 1.4))
    for i, r in enumerate(raros):
        ax.add_patch(plt.Rectangle((i, 0), 0.55, 1 if r else 0.55,
                                   color=GUINDA if r else GRIS_CLARO))
    ax.set_xlim(-0.5, n)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel("Secuencia temporal de estímulos")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRIS_CLARO),
                       plt.Rectangle((0, 0), 1, 1, color=GUINDA)],
              labels=["Estímulo frecuente (no objetivo)", "Estímulo raro (objetivo)"],
              frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    guardar(fig, "oddball.pdf")


# ---------------------------------------------------------------------
# 4. Sistema internacional 10-20 (vista superior)
# ---------------------------------------------------------------------
def fig_10_20():
    # coordenadas polares aproximadas: (radio normalizado, ángulo en grados, 0 = nasion)
    R = 1.0
    pos = {}
    # línea media
    for nom, r in [("Fpz", 0.8), ("Fz", 0.4), ("Cz", 0.0), ("Pz", 0.4), ("Oz", 0.8)]:
        ang = 90 if nom in ("Fpz", "Fz") else -90
        pos[nom] = (0, r * np.sign(np.sin(np.radians(ang))))
    # circunferencia exterior (r = 0.8)
    ext = {"Fp1": 108, "Fp2": 72, "F7": 144, "F8": 36, "T7": 180, "T8": 0,
           "P7": 216, "P8": -36, "O1": 252, "O2": -72}
    for nom, a in ext.items():
        pos[nom] = (0.8 * np.cos(np.radians(a)), 0.8 * np.sin(np.radians(a)))
    # anillo intermedio
    inter = {"F3": (-0.36, 0.44), "F4": (0.36, 0.44), "C3": (-0.4, 0), "C4": (0.4, 0),
             "P3": (-0.36, -0.44), "P4": (0.36, -0.44)}
    pos.update(inter)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.2, color="k"))
    ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, lw=0.6, ls=":", color=GRIS))
    ax.plot([0, 0], [-R, R], lw=0.5, ls=":", color=GRIS)
    ax.plot([-R, R], [0, 0], lw=0.5, ls=":", color=GRIS)
    # nariz y orejas
    ax.plot([-0.1, 0, 0.1], [0.99, 1.13, 0.99], color="k", lw=1.2)
    for s in (-1, 1):
        ax.add_patch(matplotlib.patches.Ellipse((s * 1.04, 0), 0.1, 0.3, fill=False, lw=1.2))
    linea_media = {"Fz", "Cz", "Pz", "Oz"}
    for nom, (x, y) in pos.items():
        col = GUINDA if nom in linea_media else "white"
        ax.add_patch(plt.Circle((x, y), 0.085, facecolor=col, edgecolor="k", lw=0.8, zorder=3))
        ax.text(x, y, nom, ha="center", va="center", fontsize=7,
                color="white" if nom in linea_media else "k", zorder=4)
    ax.text(0, 1.2, "Nasión", ha="center", fontsize=8)
    ax.text(0, -1.14, "Inión", ha="center", fontsize=8)
    ax.text(-1.18, -0.25, "A1", ha="center", fontsize=8)
    ax.text(1.18, -0.25, "A2", ha="center", fontsize=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "sistema_10_20.pdf")


# ---------------------------------------------------------------------
# 5. PSD por el método de Welch
# ---------------------------------------------------------------------
def fig_psd():
    fs = 240
    t = np.arange(0, 60, 1 / fs)
    x = ruido_rosa(t.size, fs, 10)
    x += 6 * np.sin(2 * np.pi * 10 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    x += 3 * np.sin(2 * np.pi * 60 * t)
    f, p = signal.welch(x, fs, nperseg=2 * fs)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.semilogy(f, p, color=AZUL, lw=1.1)
    ax.axvspan(0.1, 20, color=GUINDA, alpha=0.10, label="Banda de interés P300 (0.1–20 Hz)")
    ax.annotate(r"Pico $\alpha$ (10 Hz)", (10, p[np.argmin(abs(f - 10))]), (22, p.max() * 0.6),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.annotate("Interferencia de línea (60 Hz)", (60, p[np.argmin(abs(f - 60))]), (68, p.max() * 0.2),
                arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel(r"PSD ($\mu$V$^2$/Hz)")
    ax.legend(frameon=False, loc="lower left")
    guardar(fig, "psd_welch.pdf")


# ---------------------------------------------------------------------
# 6. Respuesta en frecuencia de filtros
# ---------------------------------------------------------------------
def fig_filtros():
    fs = 240
    b, a = signal.butter(4, [0.1, 20], btype="bandpass", fs=fs)
    w, h = signal.freqz(b, a, worN=8192, fs=fs)
    bn, an = signal.iirnotch(60, 30, fs)
    wn, hn = signal.freqz(bn, an, worN=8192, fs=fs)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axs[0].semilogx(w[1:], 20 * np.log10(abs(h[1:])), color=GUINDA, lw=1.3)
    axs[0].axhline(-3, color=GRIS, lw=0.6, ls="--")
    axs[0].text(0.012, -1.5, "−3 dB", fontsize=8, color=GRIS)
    axs[0].set_ylim(-60, 5)
    axs[0].set_xlim(0.01, 120)
    axs[0].set_title("(a) Pasa banda Butterworth, orden 4\n(0.1–20 Hz)")
    axs[0].set_xlabel("Frecuencia (Hz)")
    axs[0].set_ylabel("Magnitud (dB)")
    axs[1].plot(wn, 20 * np.log10(abs(hn) + 1e-12), color=AZUL, lw=1.3)
    axs[1].set_ylim(-40, 5)
    axs[1].set_xlim(0, 120)
    axs[1].set_title("(b) Filtro notch en 60 Hz\n(Q = 30)")
    axs[1].set_xlabel("Frecuencia (Hz)")
    fig.tight_layout()
    guardar(fig, "respuesta_filtros.pdf")


# ---------------------------------------------------------------------
# 7. Segmentación en épocas
# ---------------------------------------------------------------------
def fig_epocas():
    fs = 120
    dur = 3.0
    t = np.arange(0, dur, 1 / fs)
    x = ruido_rosa(t.size, fs, 4)
    onsets = np.arange(0.2, dur - 0.7, 0.175)
    objetivo = [3, 9]
    for i, o in enumerate(onsets):
        if i in objetivo:
            tt = t - o
            x += 7 * np.exp(-0.5 * ((tt - 0.33) / 0.06) ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(t, x, color=AZUL, lw=0.9)
    for i, o in enumerate(onsets):
        ax.axvline(o, color=GUINDA if i in objetivo else GRIS_CLARO, lw=0.8)
    for i in objetivo:
        o = onsets[i]
        ax.axvspan(o, o + 0.65, color=GUINDA, alpha=0.12)
        ax.text(o + 0.325, ax.get_ylim()[1] * 0.95, "época\n0–650 ms", ha="center", va="top", fontsize=7.5)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel(r"Amplitud ($\mu$V)")
    ax.text(0.99, -0.34, "Líneas: inicio de cada intensificación (guinda = fila/columna objetivo)",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=GRIS)
    guardar(fig, "segmentacion_epocas.pdf")


# ---------------------------------------------------------------------
# 8. Funciones de activación
# ---------------------------------------------------------------------
def fig_activaciones():
    z = np.linspace(-4, 4, 400)
    funcs = [
        ("Sigmoide", 1 / (1 + np.exp(-z))),
        ("Tangente hiperbólica", np.tanh(z)),
        ("ReLU", np.maximum(0, z)),
        (r"ELU ($\alpha=1$)", np.where(z > 0, z, np.exp(z) - 1)),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.9))
    for ax, (nom, y) in zip(axs, funcs):
        ax.axhline(0, color=GRIS_CLARO, lw=0.6)
        ax.axvline(0, color=GRIS_CLARO, lw=0.6)
        ax.plot(z, y, color=GUINDA, lw=1.4)
        ax.set_title(nom, fontsize=9)
        ax.set_xlabel("$z$")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    guardar(fig, "funciones_activacion.pdf")


# ---------------------------------------------------------------------
# 9. Sobreajuste: curvas de entrenamiento y validación
# ---------------------------------------------------------------------
def fig_sobreajuste():
    e = np.arange(1, 101)
    tr = 0.7 * np.exp(-e / 18) + 0.08 + 0.01 * rng.standard_normal(e.size) * np.exp(-e / 60)
    va = 0.7 * np.exp(-e / 18) + 0.14 + 0.0022 * np.clip(e - 30, 0, None) ** 1.35 / 3
    va += 0.01 * rng.standard_normal(e.size)
    mejor = np.argmin(va)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.plot(e, tr, color=AZUL, lw=1.3, label="Pérdida de entrenamiento")
    ax.plot(e, va, color=GUINDA, lw=1.3, label="Pérdida de validación")
    ax.axvline(e[mejor], color=GRIS, lw=0.8, ls="--")
    ax.text(e[mejor] + 1.5, 0.05, "Parada temprana", fontsize=8, color=GRIS)
    ax.text(70, 0.5, "Sobreajuste", fontsize=8, color=GUINDA)
    ax.set_xlabel("Épocas de entrenamiento")
    ax.set_ylabel("Pérdida")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, 1.0)
    guardar(fig, "sobreajuste.pdf")


# ---------------------------------------------------------------------
# 10. ROC y distribuciones de puntajes
# ---------------------------------------------------------------------
def fig_roc():
    from scipy.stats import norm
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axs[0]
    s = np.linspace(-4, 6, 500)
    ax.fill_between(s, norm.pdf(s, 0, 1), color=GRIS_CLARO, alpha=0.8, label="Clase negativa (no P300)")
    ax.fill_between(s, norm.pdf(s, 1.5, 1), color=GUINDA, alpha=0.45, label="Clase positiva (P300)")
    ax.axvline(0.8, color="k", lw=0.8, ls="--")
    ax.text(0.9, 0.44, "umbral", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Puntaje del clasificador")
    ax.set_title("(a) Distribuciones de puntajes")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax = axs[1]
    for d, col in [(0.0, GRIS), (0.75, AZUL), (1.5, GUINDA), (3.0, ORO)]:
        umbrales = np.linspace(8, -8, 400)
        fpr = 1 - norm.cdf(umbrales, 0, 1)
        tpr = 1 - norm.cdf(umbrales, d, 1)
        auc = norm.cdf(d / np.sqrt(2))
        ax.plot(fpr, tpr, color=col, lw=1.3, label=f"AUC = {auc:.2f}")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Sensibilidad (TPR)")
    ax.set_title("(b) Curvas ROC")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    guardar(fig, "roc_auc.pdf")


# ---------------------------------------------------------------------
# 11. ITR (fórmula de Wolpaw) para N = 36
# ---------------------------------------------------------------------
def itr_bits(p, n):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))


def fig_itr():
    n = 36
    p = np.linspace(1 / n, 1, 300)
    # tiempo por selección con el protocolo del dataset: 12 destellos x 175 ms x k repeticiones + 2.5 s de pausa
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axs[0].plot(p * 100, itr_bits(p, n), color=GUINDA, lw=1.4)
    axs[0].set_xlabel("Exactitud por carácter (%)")
    axs[0].set_ylabel("Bits por selección")
    axs[0].set_title("(a) Información por selección (N = 36)")
    reps = np.arange(1, 16)
    t_sel = 12 * 0.175 * reps + 2.5
    for pc, col in [(0.70, GRIS), (0.85, AZUL), (0.95, GUINDA)]:
        axs[1].plot(reps, itr_bits(pc, n) * 60 / t_sel, marker="o", ms=3, color=col, lw=1.2,
                    label=f"exactitud = {int(pc*100)} %")
    axs[1].set_xlabel("Repeticiones por carácter")
    axs[1].set_ylabel("ITR (bits/min)")
    axs[1].set_title("(b) ITR frente a repeticiones")
    axs[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    guardar(fig, "itr_repeticiones.pdf")


if __name__ == "__main__":
    for f in [fig_ritmos, fig_erp, fig_oddball, fig_10_20, fig_psd, fig_filtros, fig_epocas,
              fig_activaciones, fig_sobreajuste, fig_roc, fig_itr]:
        f()
        print("ok", f.__name__)
