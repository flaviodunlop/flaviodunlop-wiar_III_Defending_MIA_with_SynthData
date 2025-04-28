# Src: https://github.com/garrettj403/SciencePlots
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import os
import numpy as np


# Activate the 'science' style
plt.style.use(['science', 'no-latex'])

def plot_mia_attack_results(csv_path: str, model_name: str, output_dir: str = ".", figsize = (7, 5)):
    """
    Plots boxplot, histogram, and line plot for MIA attack results.
    
    Args:
        csv_path (str): Path to the CSV file.
        model_name (str): Name of the model (for titles and filenames).
        output_dir (str): Directory where plots will be saved.
    """
    # Load the results
    results = pd.read_csv(csv_path)
    figsize = figsize

    # Boxplot
    fig, ax = plt.subplots(figsize=figsize)
    data = [results['real_accuracy'], results['synth_accuracy']]
    ax.boxplot(data, tick_labels=['Real Accuracy', 'Synthetic Accuracy'], patch_artist=True)
    ax.set_title(f'{model_name} MIA Attack Accuracies (CV=5)')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, f"{model_name}_boxplot.png")
    plt.savefig(boxplot_path, dpi=500)
    plt.close()

    # Line plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(results['real_accuracy'], label='Real Accuracy', marker='o')
    ax.plot(results['synth_accuracy'], label='Synthetic Accuracy', marker='x')
    ax.set_title(f'{model_name} MIA Attack Accuracies per Cross Validation (CV=5)')
    ax.set_xlabel('Cross Validation Index')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

    # Set xticks
    ax.set_xticks(range(len(results)))  # positions (0,1,2,3,4)
    ax.set_xticklabels(range(1, len(results)+1))  # labels (1,2,3,4,5)

    plt.tight_layout()
    lineplot_path = os.path.join(output_dir, f"{model_name}_lineplot.png")
    plt.savefig(lineplot_path, dpi=500)
    plt.close()

     # Trade Off
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(results['f1_ml_real'], label='F1 TRTR', marker='o')
    ax.plot(results['f1_ml_synth'], label='F1 TSTR', marker='x')
    ax.set_title(f'{model_name} ML F1-Scores per Cross Validation (CV=5)')
    ax.set_xlabel('Cross Validation Index')
    ax.set_ylabel('F1')
    ax.legend()
    ax.grid(True)

    # Set xticks
    ax.set_xticks(range(len(results)))  # positions (0,1,2,3,4)
    ax.set_xticklabels(range(1, len(results)+1))  # labels (1,2,3,4,5)

    plt.tight_layout()
    lineplot_path = os.path.join(output_dir, f"{model_name}_lineplot_ML_f1.png")
    plt.savefig(lineplot_path, dpi=500)
    plt.close()

    print(f"Plot: {lineplot_path} and {boxplot_path} saved to output dir: {output_dir}")


def plot_mia_attack_results_combined(csv_path_copula: str, csv_path_ctgan: str, output_dir: str = ".", figsize = (7, 5)):
    """
    Plots boxplot, histogram, and line plot for MIA attack results.

    Args:
        csv_path (str): Path to the CSV file.
        model_name (str): Name of the model (for titles and filenames).
        output_dir (str): Directory where plots will be saved.
    """
    # Load the results
    results_copula = pd.read_csv(csv_path_copula)
    results_ctgan = pd.read_csv(csv_path_ctgan)
    figsize = figsize

    # Line plot MIA Attack Accuracies
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(results_copula['real_accuracy'], label='Real Accuracy', marker='s')
    ax.plot(results_copula['synth_accuracy'], label='Synthetic Accuracy Copula', marker='o')
    ax.plot(results_ctgan['synth_accuracy'], label='Synthetic Accuracy CTGAN', marker='x')
    ax.set_title(f'MIA Attack Accuracies per Cross Validation (CV=5)')
    ax.set_xlabel('Cross Validation Index')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

    # Set xticks
    ax.set_xticks(range(len(results_copula)))  # positions (0,1,2,3,4)
    ax.set_xticklabels(range(1, len(results_copula)+1))  # labels (1,2,3,4,5)

    plt.tight_layout()
    lineplot_path = os.path.join(output_dir, f"Combined_lineplot.png")
    plt.savefig(lineplot_path, dpi=500)
    plt.close()

    # Trade Off
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(results_copula['f1_ml_real'], label='F1 TRTR', marker='s')
    ax.plot(results_copula['f1_ml_synth'], label='F1 TSTR Copula', marker='o')
    ax.plot(results_ctgan['f1_ml_synth'], label='F1 TSTR CTGAN', marker='x')
    ax.set_title(f'ML F1-Scores per Cross Validation (CV=5)')
    ax.set_xlabel('Cross Validation Index')
    ax.set_ylabel('F1')
    ax.legend()
    ax.grid(True)

    # Set xticks
    ax.set_xticks(range(len(results_copula)))  # positions (0,1,2,3,4)
    ax.set_xticklabels(range(1, len(results_copula)+1))  # labels (1,2,3,4,5)

    plt.tight_layout()
    lineplot_path = os.path.join(output_dir, f"Combined_lineplot_ML_f1.png")
    plt.savefig(lineplot_path, dpi=500)
    plt.close()

    print(f"Combined Plot: {lineplot_path} saved to output dir: {output_dir}") 


if __name__ == "__main__":
    plot_mia_attack_results("../results/MIA_Attack_Results_CTGAN.csv", "CTGAN")
    plot_mia_attack_results("../results/MIA_Attack_Results_Copula.csv", "GaussianCopulaSynthesizer")
    plot_mia_attack_results_combined("../results/MIA_Attack_Results_Copula.csv", "../results/MIA_Attack_Results_CTGAN.csv")
