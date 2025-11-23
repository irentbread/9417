import matplotlib.pyplot as plt
import numpy as np

def plot_model_vs_baseline_vs_actual(y_t, y_pred_lstm, y_pred_baseline, pollutant_names, horizon, hours):

    for i, name in enumerate(pollutant_names):

        yt = y_t[:, i]
        yp_lstm = y_pred_lstm[:, i]
        yp_base = y_pred_baseline[:, i]

        plt.figure(figsize=(14, 7))

        plt.plot(yt[:hours], label="Actual", linewidth=1.2)
        plt.plot(yp_lstm[:hours], label="LSTM Prediction", linewidth=1.2)
        plt.plot(yp_base[:hours], label="Baseline Prediction", linewidth=1.2, linestyle="--")

        plt.title(f"{name} – Actual vs Predictions (LSTM vs Baseline) T={horizon}")
        plt.xlabel("Hours")
        plt.ylabel(f"{name} Concentration")
        plt.legend()

        plt.tight_layout()
        plt.show()
