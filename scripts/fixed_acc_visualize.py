#!/usr/bin/env python3
"""
Create subplot grid with individual method plots plus a combined summary plot
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def _load_and_process_data(json_path):
    """Load and process the TopK accuracy data"""

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Calculate overall accuracy
    max_k = 0
    all_methods = list(results[0]["evaluation_res"].keys())
    for question_data in results:
        qid = question_data["qid"]
        for method, acc_list in question_data["evaluation_res"].items():
            if acc_list:
                max_k = max(max_k, len(acc_list))

    overall_accuracy = {}

    for method in all_methods:
        method_accuracies = []

        for k in range(1, max_k + 1):
            correct_count = 0
            total_count = 0

            for question_data in results:
                qid = question_data["qid"]
                question_data = question_data["evaluation_res"]
                if method in question_data and question_data[method]:
                    if type(question_data[method][0]) == dict:
                        acc_list = [_["correctness"] for _ in question_data[method]]
                    else:
                        acc_list = question_data[method]
                    if k <= len(acc_list):
                        total_count += 1
                        if acc_list[k-1]:
                            correct_count += 1

            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 0.0

            method_accuracies.append(accuracy)

        overall_accuracy[method] = method_accuracies

    return overall_accuracy

def load_and_process_data(json_path):
    """Load and process the TopK accuracy data"""

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Calculate overall accuracy
    first_qid = list(results.keys())[0]
    all_methods = list(results[first_qid].keys())

    max_k = 0
    for qid, question_data in results.items():
        for method, acc_list in question_data.items():
            if acc_list:
                max_k = max(max_k, len(acc_list))

    overall_accuracy = {}

    for method in all_methods:
        method_accuracies = []

        for k in range(1, max_k + 1):
            correct_count = 0
            total_count = 0

            for qid, question_data in results.items():
                if method in question_data and question_data[method]:
                    acc_list = question_data[method]
                    if k <= len(acc_list):
                        total_count += 1
                        if acc_list[k-1]:
                            correct_count += 1

            if total_count > 0:
                accuracy = correct_count / total_count
            else:
                accuracy = 0.0

            method_accuracies.append(accuracy)

        overall_accuracy[method] = method_accuracies

    return overall_accuracy

def create_enhanced_subplot_grid_with_summary(overall_accuracy, stepconf_acc, save_path, compare_accuracy=None):
    """Create subplot grid with individual methods plus a summary plot"""

    method_display_names = {
        'majority': 'Majority Voting',
        'mean_confidence_weighted': 'Mean Confidence Weighted',
        'tail_confidence_weighted': 'Tail Confidence Weighted',
        'bottom_window_weighted': 'Bottom Window Weighted',
        'min_window_weighted': 'Min Window Weighted',
        'top10_tail_filtered': 'Top10 Tail Filtered',
        'top10_bottom_window_filtered': 'Top10 Bottom Window Filtered'
    }

    # Define colors for each method (consistent across all plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Create 4x2 subplot grid (7 individual + 1 summary)
    fig, axes = plt.subplots(4, 3, figsize=(16, 20))
    axes = axes.flatten()

    method_names = list(overall_accuracy.keys())

    # Plot individual methods in first 7 subplots
    for i, method in enumerate(method_names):
        ax = axes[i]
        accuracies = overall_accuracy[method]
        if compare_accuracy:
            compare_accuracies = compare_accuracy[method]
        k_values = list(range(1, len(accuracies) + 1))

        # Plot with consistent styling
        ax.plot(k_values, accuracies,
                color=colors[i % len(colors)],
                marker='o',
                markersize=2,
                linewidth=2,
                alpha=0.8,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=colors[i % len(colors)])
        if compare_accuracy:
            ax.plot(k_values, compare_accuracies,
                    color="black",
                    marker='o',
                    markersize=1,
                    linewidth=1,
                    alpha=0.8,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    markeredgecolor="black")

        # Fill area under curve
        ax.fill_between(k_values, accuracies, alpha=0.2, color=colors[i % len(colors)])

        # Add stepconf_acc horizontal line if available
        if method in stepconf_acc:
            stepconf_value = stepconf_acc[method]

            # Draw horizontal line for stepconf_acc
            ax.axhline(y=stepconf_value, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'{MODE}: {stepconf_value:.1%}')

            # Find minimum K where overall_acc reaches stepconf_acc
            min_k_for_stepconf = None
            for k_idx, acc in enumerate(accuracies):
                if acc >= stepconf_value:
                    min_k_for_stepconf = k_idx + 1  # k_idx is 0-based, K is 1-based
                    break
            
            max_k_for_stepconf = None
            for k_idx in range(len(accuracies)-1,-1,-1):
                acc = accuracies[k_idx]
                if acc <= stepconf_value:
                    max_k_for_stepconf = k_idx + 1
                    break 

            # Add annotation for the intersection point
            if min_k_for_stepconf is not None:
                ax.plot(min_k_for_stepconf, stepconf_value, 'ro', markersize=5, markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
                ax.annotate(f'K≥{min_k_for_stepconf}',
                           xy=(min_k_for_stepconf, stepconf_value),
                           xytext=(min_k_for_stepconf + 1, stepconf_value + 0.02),
                           fontsize=10, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.6),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            else:
                # If stepconf_acc is never reached, annotate that
                ax.text(0.98, stepconf_value, 'Not Reached',
                       transform=ax.get_yaxis_transform(),
                       horizontalalignment='right',
                       fontsize=9, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.6))
            
            if max_k_for_stepconf is not None:
                ax.plot(max_k_for_stepconf, stepconf_value, 'ro', markersize=5, markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
                ax.annotate(f'K≤{max_k_for_stepconf}',
                           xy=(max_k_for_stepconf, stepconf_value),
                           xytext=(max_k_for_stepconf + 1, stepconf_value - 0.06),
                           fontsize=10, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.6),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Styling
        display_name = method_display_names.get(method, method)
        ax.set_title(display_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('K (Top-K Traces)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)

        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add final accuracy annotation
        final_acc = accuracies[-1]
        max_acc = max(accuracies)
        if compare_accuracy:
            compare_final_acc = compare_accuracies[-1]
            compare_max_acc = max(compare_accuracies)
            final_acc_text = f'Final: {final_acc:.1%}/{compare_final_acc:.1%}, Max: {max_acc:.1%}/{compare_max_acc:.1%}'
        else:
            final_acc_text = f'Final: {final_acc:.1%}, Max: {max_acc:.1%}'
        ax.text(0.02, 0.98, final_acc_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                fontsize=10, fontweight='bold')

        # Add stepconf info to final accuracy annotation if available
        if method in stepconf_acc:
            stepconf_value = stepconf_acc[method]
            # Find minimum K for stepconf
            min_k_for_stepconf = len(accuracies)
            for k_idx, acc in enumerate(accuracies):
                if acc >= stepconf_value:
                    min_k_for_stepconf = k_idx + 1
                    break

            if min_k_for_stepconf is not None:
                print(f"{method}, {min_k_for_stepconf}")
                stepconf_text = f'{MODE}: {stepconf_value:.1%} (K≥{min_k_for_stepconf})'
            else:
                stepconf_text = f'{MODE}: {stepconf_value:.1%} (Not Reached)'

            ax.text(0.02, 0.88, stepconf_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.9),
                    fontsize=9, fontweight='bold')

        # Set consistent y-axis range for better comparison
        ax.set_ylim(0, 1)

    # Create enhanced summary plot in the last subplot
    summary_ax = axes[-1]

    # Enhanced styling for summary plot
    enhanced_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    line_styles = ['-', '-', '-', '-', '-', '-', '-']  # Use solid lines for clarity
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    line_widths = [3, 2.5, 2.5, 2.5, 2.5, 3.5, 4]  # Emphasize top performers

    # Sort methods by final accuracy for better visual hierarchy
    sorted_methods = sorted(method_names,
                          key=lambda m: overall_accuracy[m][-1],
                          reverse=True)

    # Plot with enhanced styling
    for i, method in enumerate(sorted_methods):
        accuracies = overall_accuracy[method]
        k_values = list(range(1, len(accuracies) + 1))

        orig_idx = method_names.index(method)
        display_name = method_display_names.get(method, method)

        # Shorter names for legend
        short_names = {
            'Majority Voting': 'Majority',
            'Mean Confidence Weighted': 'Mean Conf.',
            'Tail Confidence Weighted': 'Tail Conf.',
            'Bottom Window Weighted': 'Bottom Win.',
            'Min Window Weighted': 'Min Win.',
            'Top10 Tail Filtered': 'Top10 Tail',
            'Top10 Bottom Window Filtered': 'Top10 Bottom'
        }
        short_name = short_names.get(display_name, display_name)

        # Enhanced alpha for better performance methods
        alpha_value = 1.0 if i < 3 else 0.8  # Highlight top 3 performers

        summary_ax.plot(k_values, accuracies,
                       color=enhanced_colors[orig_idx % len(enhanced_colors)],
                       linestyle=line_styles[orig_idx % len(line_styles)],
                       marker=markers[orig_idx % len(markers)],
                       markersize=2,
                       linewidth=line_widths[orig_idx % len(line_widths)],
                       label=short_name,
                       alpha=alpha_value,
                       markevery=3,  # Show markers every 3 points
                       markerfacecolor='white',
                       markeredgewidth=2,
                       markeredgecolor=enhanced_colors[orig_idx % len(enhanced_colors)])

    # Enhanced styling for summary plot
    summary_ax.set_title('All Methods Comparison', fontsize=14, fontweight='bold', pad=15)
    summary_ax.set_xlabel('K (Top-K Traces)', fontsize=12, fontweight='bold')
    summary_ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')

    # Format y-axis as percentages with better range
    all_accuracies = [acc for accs in overall_accuracy.values() for acc in accs]
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    y_margin = (max_acc - min_acc) * 0.1
    summary_ax.set_ylim(max(0, min_acc - y_margin), min(1, max_acc + y_margin))

    summary_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Enhanced grid
    summary_ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
    summary_ax.spines['top'].set_visible(False)
    summary_ax.spines['right'].set_visible(False)
    summary_ax.spines['left'].set_linewidth(1.5)
    summary_ax.spines['bottom'].set_linewidth(1.5)

    # Add stepconf_acc horizontal lines to summary plot if available
    if stepconf_acc:
        for method in sorted_methods:
            if method in stepconf_acc:
                stepconf_value = stepconf_acc[method]
                orig_idx = method_names.index(method)

                # Draw horizontal line for this method's stepconf_acc
                summary_ax.axhline(y=stepconf_value, color=enhanced_colors[orig_idx % len(enhanced_colors)],
                                  linestyle=':', linewidth=1.5, alpha=0.6)

                # Find minimum K where overall_acc reaches stepconf_acc
                accuracies = overall_accuracy[method]
                min_k_for_stepconf = None
                for k_idx, acc in enumerate(accuracies):
                    if acc >= stepconf_value:
                        min_k_for_stepconf = k_idx + 1
                        break

                # Add small marker at intersection point
                if min_k_for_stepconf is not None:
                    summary_ax.plot(min_k_for_stepconf, stepconf_value,
                                   marker='s', markersize=4,
                                   color=enhanced_colors[orig_idx % len(enhanced_colors)],
                                   markerfacecolor='white',
                                   markeredgewidth=1.5)

    # Enhanced legend with performance annotations
    legend = summary_ax.legend(loc='lower right', fontsize=10, frameon=True,
                              fancybox=True, shadow=True, ncol=2,
                              columnspacing=0.5, handlelength=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # Add performance summary text box (updated to include stepconf info)
    performance_text = []
    for method in sorted_methods[:3]:  # Top 3 performers
        final_acc = overall_accuracy[method][-1]
        short_name = short_names.get(method_display_names.get(method, method), method)

        # Add stepconf info if available
        if method in stepconf_acc:
            stepconf_value = stepconf_acc[method]
            accuracies = overall_accuracy[method]
            min_k_for_stepconf = None
            for k_idx, acc in enumerate(accuracies):
                if acc >= stepconf_value:
                    min_k_for_stepconf = k_idx + 1
                    break

            if min_k_for_stepconf is not None:
                performance_text.append(f"{short_name}: {final_acc:.1%} (K≥{min_k_for_stepconf} for {stepconf_value:.1%})")
            else:
                performance_text.append(f"{short_name}: {final_acc:.1%} ({MODE} {stepconf_value:.1%} not reached)")
        else:
            performance_text.append(f"{short_name}: {final_acc:.1%}")

    text_content = "Top Performers:\n" + "\n".join(performance_text)
    summary_ax.text(0.02, 0.98, text_content,
                   transform=summary_ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                   fontsize=9, fontweight='bold')

    # Add overall title
    fig.suptitle('TopK Accuracy Analysis: Individual Methods and Overall Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Saved enhanced subplot grid with summary: {save_path}')

def get_stepconf_acc(json_path):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics_map = ["majority", "mean_confidence_weighted", "tail_confidence_weighted", "bottom_window_weighted", "min_window_weighted", "top10_tail_filtered", "top10_bottom_window_filtered"]
    acc = {}
    # Count entries where majority's correct is true
    for metric in metrics_map:
        count = []
        for entry in data:
            if not entry['evaluation_res'].get(metric, 0):
                continue
            if entry['evaluation_res'][metric]['correct'] == True:
                count.append(1)
            else:
                count.append(0)
        correct_num = sum(count)
        acc[metric] = correct_num/len(count)
    return acc

MODE="SelfStepConf"
# MODE="StepConf"
def main():
    res_dir = "/share/yangxizhong/output/deepconf/stepconf_deepthink/gpqad-low-warmupK-parallel-20251202"
    print('Loading and processing data...')
    topK_json_path = f"{res_dir}/fixed_acc_test_H800_stepbased_Alltop10_20251204.json"
    overall_accuracy = _load_and_process_data(topK_json_path)
    compare_accuracy = _load_and_process_data("/share/yangxizhong/output/deepconf/baseline-dpsk/gpqad-B512/fixed_acc_test_H800_windowbased_Alltop10_20251124.json")
    # compare_accuracy = None
    
    # if len(compare_accuracy["majority"]) != len(overall_accuracy["majority"]):
    #     for key, values in overall_accuracy.items():
    #         overall_accuracy[key].append(compare_accuracy[key][0])

    print('Loading stepconf acc...')
    if MODE=="SelfStepConf":
        # stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-95-40-80-20251021/hmmt2025_dpsk_online.json"
        # stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/hmmt2025-95-80-80-20251107/wait_with_downward/wait_with_downward_dpsk_online.json"
        stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/gpqad-95-80-80-20251107/gpqad_dpsk_online.json"
        # stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2024-95-80-80-20251107/aime2024_dpsk_online.json"
        # stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/aime2025-95-80-80-20251107/aime2025_dpsk_online.json"
        # stepconf_json_path = "/share/yangxizhong/output/deepconf/selfstepconf_deepthink/brumo2025-95-80-80-20251107/brumo2025_dpsk_online.json"
    elif MODE=="StepConf":
        stepconf_json_path = "/share/yangxizhong/output/deepconf/stepconf_deepthink/hmmt2025-low-20251011/hmmt2025_dpsk_online_16x1_update.json"
    stepconf_acc = get_stepconf_acc(stepconf_json_path)

    print(f'Processing {len(overall_accuracy)} methods')

    print('Creating enhanced subplot grid with summary...')
    save_path = f"{res_dir}/fixed_acc_test_{MODE}_AllTop10_compare.png"
    create_enhanced_subplot_grid_with_summary(overall_accuracy, stepconf_acc, save_path, compare_accuracy)

    print('Enhanced plot with summary generated successfully!')

if __name__ == '__main__':
    main()