# %% Create 2x2 residual diagnostic panel

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
# Update these paths to your actual exported single-panel figures
fig_dir = os.path.join(base_dir, "output", "actual", "resid_plots")

panel_files = {
    "a": os.path.join(fig_dir, "standardized_residuals_vs_fitted.png"),
    "b": os.path.join(fig_dir, "aggregate_monthly_mean_residual.png"),
    "c": os.path.join(fig_dir, "acf_aggregate_monthly_mean_residuals.png"),
    "d": os.path.join(fig_dir, "residual_event_time_check_maxs25.png"),
}

output_path = os.path.join(fig_dir, "model_diagnostics.png")

panel_titles = {
    "a": "(a) Standardized residuals vs fitted values",
    "b": "(b) Aggregate monthly mean residual",
    "c": "(c) ACF of aggregate monthly mean residuals",
    "d": "(d) Event-time residual check: maxs ≥ 25",
}

# Read images
imgs = {k: mpimg.imread(v) for k, v in panel_files.items()}

# Create 2x2 panel
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, key in zip(axes.ravel(), ["a", "b", "c", "d"]):
    ax.imshow(imgs[key])
    ax.set_title(panel_titles[key], fontsize=14, pad=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved combined diagnostic panel to: {output_path}")