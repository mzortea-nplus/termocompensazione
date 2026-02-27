"""
src/plot.py
Genera i grafici per tutti i modelli eseguiti.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List

from src.models import ModelResult
from src.config import AppConfig


def plot_results(
    results: List[ModelResult],
    df: pd.DataFrame,
    cfg: AppConfig,
    filename: str,
) -> None:
    """
    Per ogni risultato:
      - subplot superiore : dati reali vs predizione
      - subplot inferiore : residui (predizione – reale)
    """
    if not results:
        print("[plot] Nessun risultato da visualizzare.")
        return

    time_col = cfg.data.time_column

    # Assicura la directory di output esista (se richiesto)
    if cfg.output.save_plots:
        Path(cfg.output.plot_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Grafico comparativo tutti i modelli ─────────────────────────────
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]  # normalizza a lista di coppie
    df = df.sort_values(by=time_col)

    fig.suptitle("Model vs Data  |  Residui", fontsize=13, fontweight="bold")

    for i, res in enumerate(results):
        ax_fit, ax_res = axes[i]

        # — Fit ——————————————————————————————————————
        ax_fit.plot(
            df[time_col],
            df[res.target_col],
            color="black",
            lw=1,
            label="Dati reali",
            alpha=0.7,
        )
        ax_fit.plot(res.t, res.y_pred, color="steelblue", lw=1.5, label="Predizione")
        ax_fit.set_title(f"{res.name}  |  R²={res.r2:.3f}  RMSE={res.rmse:.5f}")
        ax_fit.set_xlabel("Tempo")
        ax_fit.set_ylabel(res.target_col)
        ax_fit.legend(fontsize=8)
        ax_fit.grid(alpha=0.3)

        # — Residui ———————————————————————————————————
        residuals = res.y_pred - res.y_true
        ax_res.plot(res.t, residuals, color="tomato", lw=1, alpha=0.8)
        ax_res.axhline(0, color="black", lw=0.8, linestyle="--")
        ax_res.set_title(f"Residui – {res.name}")
        ax_res.set_xlabel("Tempo")
        ax_res.set_ylabel("Pred − Reale")
        ax_res.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, f"comparison_{filename}", cfg)

    # ── 2. Overlay tutti i modelli su un unico grafico ─────────────────────
    if n > 1:
        fig2, ax = plt.subplots(figsize=(14, 4))
        ax.plot(
            df[time_col],
            df[results[0].target_col],
            color="black",
            lw=1,
            label="Dati reali",
            alpha=0.7,
        )
        colors = plt.cm.tab10.colors
        for i, res in enumerate(results):
            ax.plot(
                res.t,
                res.y_pred,
                color=colors[i % len(colors)],
                lw=1.5,
                label=f"{res.name}  R²={res.r2:.3f}",
            )
        ax.set_title("Overlay modelli")
        ax.set_xlabel("Tempo")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig2, f"overlay_{filename}", cfg)


def _save_or_show(fig: plt.Figure, name: str, cfg: AppConfig) -> None:
    if cfg.output.save_plots:
        out = Path(cfg.output.plot_dir) / f"{name}.png"
        fig.savefig(out, dpi=150)
        print(f"[plot] Salvato: {out}")
    if cfg.output.show_plots:
        plt.show()
    plt.close(fig)
