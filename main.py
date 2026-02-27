"""
main.py
Entry point del progetto di regressione.

Uso:
    python main.py                    # usa config.yaml nella directory corrente
    python main.py --config my.yaml   # specifica un file di config diverso
"""

from __future__ import annotations
import argparse
import sys

from src.config import load_config
from src.data import load_data
from src.models import run_models
from src.plot import plot_results


def main(config_path: str = "config.yaml") -> None:
    print("=" * 60)
    print("  Regression Pipeline")
    print("=" * 60)

    # 1. Carica configurazione
    print(f"\n[config] Caricamento da: {config_path}")
    cfg = load_config(config_path)

    # 2. Carica dati
    print("\n[data] Caricamento dati...")
    df, tmp_sensors, sensors = load_data(cfg)

    for s in sensors:
        try:
            print(f"\n[data] Caricamento dati per {s}...")
            df_s = df[df[s].notna()]
            # 3. Esegui modelli
            print("\n[models] Esecuzione modelli...")
            results = run_models(df_s, tmp_sensors, [s], cfg, filename=s)
            if not results:
                print(
                    "\n[ATTENZIONE] Nessun algoritmo abilitato nel config. "
                    "Imposta almeno uno tra linear_regression e multiple_linear_regression a true."
                )
                sys.exit(1)
        except Exception as e:
            print(f"\n[data] Errore: {e}")
            continue

        # 4. Stampa riepilogo
        print("\n── Riepilogo risultati ─────────────────────────────────")
        for r in results:
            print(f"  {r.summary()}")

        # 5. Plot
        if cfg.output.show_plots or cfg.output.save_plots:
            print("\n[plot] Generazione grafici...")
            plot_results(results, df_s, cfg, filename=s)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Percorso al file di configurazione YAML (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
