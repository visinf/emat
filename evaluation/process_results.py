import json
import numpy as np
import csv
import matplotlib.pyplot as plt
plt.rc("font", family="serif")


def save_main_table(res_path, out_name):
    table = [["Dataset", "Method", 
              "Original Acc.", "Original mIoU", 
              "Partially Augmented Acc.", "Partially Augmented mIoU", 
              "Fully Augmented Acc.", "Fully Augmented mIoU"]]

    for dataset in ["pascal", "coco"]:
        for method in ["cst", "emat"]:
            row = [dataset.upper(), method.upper()]
            for setting in ["original", "partially-augmented", "fully-augmented"]:
                acc = []
                miou = []
                for f in range(4):
                    file = f"{res_path}/{method}_{dataset}_{setting}_fold-{f}_2w-1s.json"
                    with open(file, "r", encoding="utf-8") as file_info:
                        data = json.load(file_info)
                    acc.append(data["test/er"])
                    miou.append(data["test/miou"])
                row.append(f"{np.mean(acc):.2f}")
                row.append(f"{np.mean(miou):.2f}")
            table.append(row)

    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    

def save_ablation_table(res_path, out_name):
    table = [["Method", 
              "All Dataset Acc.", "All Dataset mIoU",
              "Small Objects Acc.", "Small Objects mIoU"]]

    for method in ["cst", "cst-large", "emat"]:
        row = [method.upper()]
        for eval_interval in ["all", "small"]:
            acc = []
            miou = []
            for f in range(4):
                file = f"{res_path}/{method}_pascal_original_fold-{f}_"
                if eval_interval == "all":
                    file += "2w-1s.json"
                else:
                    file += "1w-1s_no-empty-masks_interval-0-15.json"
                with open(file, "r", encoding="utf-8") as file_info:
                    data = json.load(file_info)
                acc.append(data["test/er"])
                miou.append(data["test/miou"])
            row.append(f"{np.mean(acc):.2f}")
            row.append(f"{np.mean(miou):.2f}")
        table.append(row)

    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table)
 
 
def save_analysis_small_objects(res_path, out_name):
    def plot_grouped_bars(data, out_name):
        ticks_font = 11.5
        labels_font = 12.5
        title_font = 14
        
        colors = [
            "#C9DAF8",
            "#FCE5CD",
            "#D9EAD3"
        ]

        intervals = ["0 - 5", "5 - 10", "10 - 15"]
        n = len(intervals)
        bar_height = 1.0
        group_gap = 1.0
        
        # Bar positions and Y ticks
        acc_bars = np.arange(n)
        miou_bars = np.arange(n) + n + group_gap
        yticks = np.concatenate([acc_bars, miou_bars])
        yticklabels = intervals + intervals

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=False)
        for i, dataset in enumerate(data.keys()):
            ax = axes[i]

            acc_cst = np.array(data[dataset]["cst"]["acc"]) 
            miou_cst = np.array(data[dataset]["cst"]["miou"]) 
            acc_impr = np.array(data[dataset]["emat"]["acc"]) - acc_cst
            miou_impr = np.array(data[dataset]["emat"]["miou"]) - miou_cst

            # Bars for accuracy
            acc = ax.barh(acc_bars, acc_cst, height=bar_height, color=colors[0], 
                          edgecolor="black", label="CST$^{*}$ Acc.")
            acc_imp = ax.barh(acc_bars, acc_impr, height=bar_height, 
                              left=acc_cst, color=colors[2], edgecolor="black", 
                              label="EMAT Improvement")

            # Bars for mIoU
            miou = ax.barh(miou_bars, miou_cst, height=bar_height, color=colors[1], 
                           edgecolor="black", label="CST$^{*}$ mIoU")
            miou_imp = ax.barh(miou_bars, miou_impr, height=bar_height, 
                               left=miou_cst, color=colors[2], edgecolor="black")

            # Annotations inside baseline bars
            for bars, values in zip([acc, miou], [acc_cst, miou_cst]):
                for bar, val in zip(bars, values):
                    ax.text(val-0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}", 
                            ha="right", va="center", fontsize=ticks_font, color="black")

            # Annotations for improvements outside bars
            for bars, values in zip([acc_imp, miou_imp], [acc_impr, miou_impr]):
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() + 0.5, 
                            bar.get_y() + bar.get_height()/2,
                            f"+{val:.1f}", ha="left", va="center", fontsize=ticks_font, 
                            color="black")

            # Ticks and labels
            ax.set_yticks(yticks)
            if i == 0:
                ax.set_yticklabels(yticklabels, fontsize=ticks_font)
                ax.set_ylabel("Object Size (%)", fontsize=labels_font)
                ax.yaxis.set_tick_params(labelsize=ticks_font)
            else:
                ax.set_yticklabels([])
                ax.yaxis.set_tick_params(length=0)
            ax.invert_yaxis()
            ax.set_xlabel("Percentage (%)", fontsize=labels_font)
            min_val = min(np.min(acc_cst), np.min(miou_cst)) - 7
            ax.set_xlim(max(min_val, 0), 100)
            if dataset == "pascal":
                title = "PASCAL-$5^{i}$"
            else:
                title = "COCO-$20^{i}$"
            ax.set_title(title, fontsize=title_font)
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

        # Legend
        handles, labels = axes[0].get_legend_handles_labels()
        legends = ["CST$^{*}$ Acc.", "CST$^{*}$ mIoU", "EMAT Improvement"]
        by_label = dict(zip(labels, handles))
        ordered_handles = [by_label[label] for label in legends if label in by_label]
        ordered_labels = [label for label in legends if label in by_label]

        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            fontsize=labels_font,
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            bbox_transform=fig.transFigure
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_name, bbox_inches="tight", dpi=300)
        plt.close()

    plot_data = {}
    for dataset in ["pascal", "coco"]:
        plot_data[dataset] = {}
        for method in ["cst", "emat"]:
            plot_data[dataset][method] = {"acc": [], "miou": []}
            f_name = f"{res_path}/{method}_{dataset}_original_fold-"
            for interval in ["0-5", "5-10", "10-15"]:
                acc = []
                miou = []
                for f in range(4):
                    file = f"{f_name}{f}_1w-1s_no-empty-masks_interval-{interval}.json"
                    with open(file, "r", encoding="utf-8") as file_info:
                        data = json.load(file_info)
                    acc.append(data["test/er"])
                    miou.append(data["test/miou"])
                plot_data[dataset][method]["acc"].append(np.mean(acc))
                plot_data[dataset][method]["miou"].append(np.mean(miou))
    plot_grouped_bars(plot_data, out_name)
                  
        
def save_fss_results(res_path, out_name):
    table = [["Dataset",  "1-way 1-shot", "1-way 5-shot"]]

    for dataset in ["pascal", "coco"]:
        row = [dataset.upper()]
        for shot in [1, 5]:
            miou = []
            f_name = f"{res_path}/ematseg_{dataset}_original_fold-"
            for f in range(4):
                file = f"{f_name}{f}_1w-{shot}s_only-seg_no-empty-masks.json"
                with open(file, "r", encoding="utf-8") as file_info:
                    data = json.load(file_info)
                miou.append(data["test/miou"])
            row.append(f"{np.mean(miou):.1f}")
        table.append(row)

    with open(out_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    

   
if __name__ == "__main__":
    RESULTS_PATH = "../results"
    save_main_table(RESULTS_PATH, "main_table.csv")
    save_analysis_small_objects(RESULTS_PATH, "analysis_small_objects.pdf")
    save_ablation_table(RESULTS_PATH, "ablation_table.csv")
    save_fss_results(RESULTS_PATH, "fss_table.csv")