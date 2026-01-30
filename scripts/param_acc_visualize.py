#!/usr/bin/env python3
"""
Visualize parameter accuracy curves for each question.
Shows how different methods perform across different parameter values.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import Dict, List
import math

def get_question_parm_acc(base_dir, param_dir, res_filename):
    """Load results from different parameter configurations"""
    all_results = {}
    for k, d in param_dir.items():
        res_path = os.path.join(base_dir, d, res_filename)
        if os.path.exists(res_path):
            with open(res_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_results[k] = data
                    print(f"Loaded: {res_path} ({len(data)} entries)")
                except json.JSONDecodeError:
                    print(f"[Warning] JSON decode error in: {res_path}")
        else:
            print(f"[Warning] File not found: {res_path}")
    return all_results

def process_acc_byparam(results):
    """Process results to get param_acc structure"""
    param_acc = {}

    for param, param_res in results.items():
        for question_res in param_res:
            qid = question_res["qid"]
            if qid not in param_acc:
                param_acc[qid] = {}
            for method, method_res in question_res["evaluation_res"].items():
                correct = method_res.get("correct")
                if method not in param_acc[qid]:
                    param_acc[qid][method] = {param: correct}
                else:
                    param_acc[qid][method][param] = correct

    return param_acc

def plot_param_accuracy_curves(param_acc, save_dir):
    """
    Plot parameter accuracy curves for each question.
    Each question gets its own subplot showing all methods.
    """

    # Method display names and enhanced colors
    method_display_names = {
        'majority': 'Majority',
        'mean_confidence_weighted': 'Mean Conf.',
        'tail_confidence_weighted': 'Tail Conf.',
        'bottom_window_weighted': 'Bottom Win.',
        'min_window_weighted': 'Min Win.',
        'top10_tail_filtered': 'Top10 Tail',
        'top10_bottom_window_filtered': 'Top10 Bottom'
    }

    # Enhanced color palette with better contrast
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#E67E22']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    line_styles = ['-', '-', '-', '-', '-', '-', '-']  # Use solid lines for clarity

    # Get all questions and sort them
    question_ids = sorted(param_acc.keys(), key=int)
    num_questions = len(question_ids)

    print(f"Creating plots for {num_questions} questions")

    # Calculate grid size for subplots with better spacing
    cols = 5  # Increased columns for better aspect ratio
    rows = math.ceil(num_questions / cols)

    # Create figure with enhanced size and spacing
    fig, axes = plt.subplots(rows, cols, figsize=(25, 4 * rows))

    # Set background color
    fig.patch.set_facecolor('white')

    # Handle case where we have only one row
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif num_questions == 1:
        axes = np.array([[axes]])

    axes = axes.flatten()

    # Get all methods from first question to maintain consistency
    first_qid = question_ids[0]
    all_methods = list(param_acc[first_qid].keys())

    # Plot each question
    for idx, qid in enumerate(question_ids):
        ax = axes[idx]
        question_data = param_acc[qid]

        # Plot each method for this question
        for method_idx, method in enumerate(all_methods):
            if method not in question_data:
                continue

            method_data = question_data[method]

            # Extract parameter values and accuracies
            params = sorted(method_data.keys(), key=int)
            accuracies = [1.0 if method_data[p] else 0.0 for p in params]
            param_values = [int(p) for p in params]

            # Plot the curve with enhanced styling
            display_name = method_display_names.get(method, method)
            ax.plot(param_values, accuracies,
                   color=colors[method_idx % len(colors)],
                   marker=markers[method_idx % len(markers)],
                   linestyle=line_styles[method_idx % len(line_styles)],
                   linewidth=2.5,
                   markersize=7,
                   label=display_name,
                   alpha=0.9,
                   markerfacecolor='white',
                   markeredgewidth=2,
                   markeredgecolor=colors[method_idx % len(colors)])

        # Enhanced styling for each subplot
        ax.set_title(f'Q{qid}', fontsize=13, fontweight='bold', pad=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
        ax.set_xlabel('Parameter α', fontsize=11, fontweight='bold')
        ax.set_ylabel('Result', fontsize=11, fontweight='bold')

        # Enhanced y-axis with better labels
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['✗ Wrong', '✓ Correct'], fontsize=10)

        # Enhanced grid
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Set x-axis ticks with better spacing
        if param_values:
            ax.set_xlim(min(param_values) - 3, max(param_values) + 3)
            ax.set_xticks(param_values)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)

        # Add subtle background color for better contrast
        ax.set_facecolor('#fafafa')

    # Hide unused subplots
    for idx in range(num_questions, len(axes)):
        axes[idx].set_visible(False)

    # Create a separate legend subplot
    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    legend_ax.grid(False)

    # Create legend elements
    legend_elements = []
    for method_idx, method in enumerate(all_methods):
        display_name = method_display_names.get(method, method)
        legend_elements.append(plt.Line2D([0], [0],
                                        color=colors[method_idx % len(colors)],
                                        marker=markers[method_idx % len(markers)],
                                        linewidth=2.5,
                                        markersize=8,
                                        label=display_name,
                                        markerfacecolor='white',
                                        markeredgewidth=2))

    # Add enhanced legend
    legend = legend_ax.legend(handles=legend_elements,
                             loc='upper center',
                             bbox_to_anchor=(0.5, -0.02),
                             ncol=4,
                             fontsize=12,
                             frameon=True,
                             fancybox=True,
                             shadow=True,
                             columnspacing=1.5,
                             handlelength=2.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # Add overall title with better styling
    fig.suptitle('Parameter Sensitivity Analysis: Method Performance Across Questions',
                 fontsize=18, fontweight='bold', y=0.98)

    # Adjust layout with better spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.15,
                       hspace=0.4, wspace=0.3)

    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'enhanced_param_accuracy_curves_by_question.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved enhanced parameter accuracy curves plot: {save_path}")
    return save_path

def plot_method_summary_curves(param_acc, save_dir):
    """
    Create a summary plot showing overall performance of each method across parameters.
    """

    method_display_names = {
        'majority': 'Majority',
        'mean_confidence_weighted': 'Mean Conf.',
        'tail_confidence_weighted': 'Tail Conf.',
        'bottom_window_weighted': 'Bottom Win.',
        'min_window_weighted': 'Min Win.',
        'top10_tail_filtered': 'Top10 Tail',
        'top10_bottom_window_filtered': 'Top10 Bottom'
    }

    # Enhanced color palette matching the main plot
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#E67E22']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    # Get all methods and parameters
    first_qid = list(param_acc.keys())[0]
    all_methods = list(param_acc[first_qid].keys())
    all_params = sorted(param_acc[first_qid][all_methods[0]].keys(), key=int)

    # Create enhanced figure
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')

    # Calculate overall accuracy for each method at each parameter
    for method_idx, method in enumerate(all_methods):
        param_accuracies = []

        for param in all_params:
            correct_count = 0
            total_count = 0

            for qid in param_acc:
                if method in param_acc[qid] and param in param_acc[qid][method]:
                    total_count += 1
                    if param_acc[qid][method][param]:
                        correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0
            param_accuracies.append(accuracy)

        param_values = [int(p) for p in all_params]
        display_name = method_display_names.get(method, method)

        # Enhanced plot styling
        line = ax.plot(param_values, param_accuracies,
                      color=colors[method_idx % len(colors)],
                      marker=markers[method_idx % len(markers)],
                      linewidth=3.5,
                      markersize=10,
                      label=display_name,
                      alpha=0.9,
                      markerfacecolor='white',
                      markeredgewidth=2.5,
                      markeredgecolor=colors[method_idx % len(colors)])

        # Add subtle shadow effect
        ax.plot(param_values, param_accuracies,
               color=colors[method_idx % len(colors)],
               linewidth=5,
               alpha=0.2,
               zorder=0)

    # Enhanced styling
    ax.set_title('Overall Method Performance Across Parameter Values',
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Parameter α', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=16, fontweight='bold')

    # Enhanced y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1)

    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Enhanced tick styling
    ax.tick_params(axis='x', labelsize=14, pad=8)
    ax.tick_params(axis='y', labelsize=14, pad=8)

    # Set background color
    ax.set_facecolor('#fafafa')

    # Enhanced legend
    legend = ax.legend(loc='best', fontsize=13, frameon=True,
                      fancybox=True, shadow=True, ncol=2,
                      columnspacing=1.5, handlelength=2.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout(pad=2)

    # Save the enhanced summary plot
    summary_path = os.path.join(save_dir, 'enhanced_method_summary_param_curves.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved enhanced method summary curves plot: {summary_path}")
    return summary_path

def main():
    """Main function to load data and create visualizations"""

    print("Loading parameter accuracy data...")

    # Configuration
    base_dir = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink"
    alpha_dir = {
        "10": "hmmt2025-95-10-80-20251021",
        "30": "hmmt2025-95-30-80-20251021",
        "40": "hmmt2025-95-40-80-20251021",
        "50": "hmmt2025-95-50-80-20251020",
        "60": "hmmt2025-95-60-80-20251021",
        "70": "hmmt2025-95-70-80-20251021",
        "90": "hmmt2025-95-90-80-20251017-v0"
    }
    res_filename = "hmmt2025_dpsk_online.json"

    # Load and process data
    all_results = get_question_parm_acc(base_dir, alpha_dir, res_filename)
    param_acc = process_acc_byparam(all_results)

    print(f"Processed data for {len(param_acc)} questions")

    # Create output directory
    output_dir = "/share/yangxizhong/output/deepconf/param_accuracy_plots"

    # Generate visualizations
    print("Creating parameter accuracy curves by question...")
    question_plot_path = plot_param_accuracy_curves(param_acc, output_dir)

    print("Creating method summary curves...")
    summary_plot_path = plot_method_summary_curves(param_acc, output_dir)

    print(f"\nGenerated plots:")
    print(f"- Question-wise curves: {question_plot_path}")
    print(f"- Method summary curves: {summary_plot_path}")

    return param_acc, question_plot_path, summary_plot_path

if __name__ == "__main__":
    main()