import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
import shap

class DrawUtil:

  # Color theme (SCI Journal Style: Blue-gray represents normal, pink represents abnormal)
  COLOR_NORMAL = "rgba(44, 130, 201, 0.85)"  # Blue-gray (0: Normal)
  COLOR_ABNORMAL = "rgba(225, 43, 136, 0.85)"  # Pink (1: Abnormal)
  COLOR_LINK_STABLE_0 = "rgba(44, 130, 201, 0.3)"  # Blue->Blue
  COLOR_LINK_STABLE_1 = "rgba(225, 43, 136, 0.3)"  # Red->Red
  COLOR_LINK_WORSEN = "rgba(255, 165, 0, 0.5)"  # Orange Warning: Blue->Red (Worse)
  COLOR_LINK_IMPROVE = "rgba(46, 204, 113, 0.5)"  # Green Hope: Red->Blue (Improvement)

  def __init__(self, config, expr_path, logger):
    self.config = config
    self.expr_path = expr_path
    self.configure_plotting_style()
    self.logger = logger
    
  def configure_plotting_style(self):
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        pass
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300
    
  def confusion_matrix(self, task_name, y, y_pred, fig_report_list, is_binary: bool = True):
    cm_title = f'OOF CM: {task_name}'
    class_names = ['Normal (0)', 'Abnormal (1)'] if is_binary else ['Monofocal (0)', 'EDOF (1)', 'Multifocal (2)']
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent, nan=0)
    
    sns.heatmap(cm_percent,  annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                annot_kws={"fontsize": 12, "fontfamily": "serif"})
    plt.title(cm_title, fontsize=14, fontfamily="serif")
    plt.tight_layout()
    cm_filename = f"cm_{task_name}_600dpi.png"
    plt.savefig(os.path.join(self.expr_path, cm_filename), dpi=600, bbox_inches='tight')
    plt.close()
    
    fig_report_list.append(f"- **Academic Interpretation**: This chart displays the model's classification performance under 5-fold cross-validation (full validation set). The values in the heatmap represent the proportion of each predicted class within the corresponding true class. This visualization provides an intuitive understanding of the model's sensitivity and specificity balance when handling this clinical task.\n")
    
  def roc_curve(self, task_name, y, y_prob, fig_report_list, is_binary: bool = True):
    if not is_binary: return
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontfamily='serif')
    plt.ylabel('True Positive Rate', fontsize=12, fontfamily='serif')
    plt.title(f'OOF ROC: {task_name}', fontsize=14, fontfamily='serif')
    plt.legend(loc="lower right", prop={'family': 'serif', 'size': 11})
    plt.tight_layout()
    roc_filename = f"roc_{task_name}_600dpi.png"
    plt.savefig(os.path.join(self.expr_path, roc_filename), dpi=600)
    plt.close()

    # Write documentation for the ROC curve
    fig_report_list.append(f"### Figure File: {roc_filename}")
    fig_report_list.append(f"- **Figure Type**: Receiver Operating Characteristic Curve (ROC Curve)")
    fig_report_list.append(f"- **Target Task**: {task_name}")
    fig_report_list.append(
        f"- **Academic Interpretation**: This chart illustrates the dynamic relationship between sensitivity (TPR) and specificity (FPR) for this binary classification task across various decision thresholds. The area under the curve (AUC) reaches {roc_auc:.3f}, directly quantifying the model's ability to distinguish between normal and abnormal patients. The closer the curve approaches the top-left corner, the higher the model's predictive performance.\n")
    
  def shap_values(self, task_name, model, X, feature_names, fig_report_list, is_binary: bool = True):
    shap_title = f"Global SHAP: {task_name}"
    shap_filename = f"shap_summary_{task_name}_600dpi.png"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    x_plot = pd.DataFrame(X, columns=feature_names)
    
    if is_binary:
      shap.summary_plot(shap_values, x_plot, show=False)
    else:
      shap.summary_plot(shap_values, x_plot, plot_type="bar", show=False)
      plt.legend(
        handles=[
            mpatches.Patch(color="#008BFB", label="Monofocal"),
            mpatches.Patch(color="#FF0051", label="EDOF"),
            mpatches.Patch(color="#7A8100", label="Multifocal")
        ],
        loc="lower right"
      )
    
    plt.title(shap_title, fontsize=14, fontfamily='serif', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(self.expr_path, shap_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Write documentation for the SHAP summary plot
    fig_report_list.append(f"### Figure File: {shap_filename}")
    fig_report_list.append(f"- **Figure Type**: Global SHAP Summary Plot")
    fig_report_list.append(f"- **Target Task**: {task_name}")
    if is_binary:
      fig_report_list.append(f"- **Academic Interpretation**: This chart quantifies the contribution of each clinical input feature to the model's final output using the SHAP algorithm. The Y-axis is ordered by the absolute importance of the features; the X-axis represents the SHAP values (positive values push the model prediction towards the abnormal class, while negative values push it towards the normal class). The color gradient of the scatter points indicates the range of original feature values from high to low, revealing nonlinear directional associations between specific clinical indicators and prognosis outcomes.\n")
    else:
      fig_report_list.append(
          f"- **Academic Interpretation**: This chart displays the overall importance ranking of clinical features in a multi-classification task. Different colored bars represent the discriminative contribution of each feature to different intraocular lens categories (monofocal, EDOF, multifocal), intuitively showcasing the intrinsic clinical logic behind personalized lens recommendations.\n")
    fig_report_list.append("-" * 60 + "\n")
    
  def get_node_link_colors(self, source, target):
    """Assign colors to links based on the direction of state transition"""
    link_colors = []
    for s, t in zip(source, target):
        if s % 2 == 0 and t % 2 == 0:  # 0 -> 0
            link_colors.append(self.COLOR_LINK_STABLE_0)
        elif s % 2 == 1 and t % 2 == 1:  # 1 -> 1
            link_colors.append(self.COLOR_LINK_STABLE_1)
        elif s % 2 == 0 and t % 2 == 1:  # 0 -> 1 (Worse)
            link_colors.append(self.COLOR_LINK_WORSEN)
        else:  # 1 -> 0 (Improvement)
            link_colors.append(self.COLOR_LINK_IMPROVE)
    return link_colors

  def plot_2_stage_sankey(self, df, task_base, title_prefix):
    """Draw a 2-stage Sankey diagram: 3-month prediction -> 6-month prediction"""
    col_3m = f"Pred_{task_base.replace('6m', '3m')}"
    col_6m = f"Pred_{task_base}"

    if col_3m not in df.columns or col_6m not in df.columns: return

    # Filter out invalid labels
    valid = df[(df[col_3m].isin([0, 1])) & (df[col_6m].isin([0, 1]))]
    if len(valid) == 0: return

    # Count the number of each state transition
    counts = valid.groupby([col_3m, col_6m]).size().to_dict()

    # Define nodes
    labels = [
        "3m Pred: Normal (0)", "3m Pred: Abnormal (1)",
        "6m Pred: Normal (0)", "6m Pred: Abnormal (1)"
    ]
    node_colors = [self.COLOR_NORMAL, self.COLOR_ABNORMAL, self.COLOR_NORMAL, self.COLOR_ABNORMAL]

    # Build links (Source: 0, 1 -> Target: 2, 3)
    source, target, value = [], [], []
    for (src_val, tgt_val), cnt in counts.items():
        source.append(int(src_val))
        target.append(int(tgt_val) + 2)
        value.append(cnt)

    link_colors = self.get_node_link_colors(source, target)

    # Create Sankey diagram object
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25, thickness=30, line=dict(color="black", width=0.5),
            label=labels, color=node_colors
        ),
        link=dict(
            source=source, target=target, value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=f"<b>Patient Trajectory: {title_prefix} (3m to 6m)</b>",
        font=dict(size=16, family="Times New Roman"), # Increase font size to adapt high-resolution
        width=1000, height=600,
        plot_bgcolor='white', paper_bgcolor='white'
    )

    # Export as high-resolution PNG (scale=6, simulate ~600 DPI)
    png_path = os.path.join(self.expr_path, f"Sankey_2Stage_{title_prefix}_600dpi.png")
    fig.write_image(png_path, scale=6)
    self.logger.info(f"  [+] Successfully generated: {png_path}")
    
  def plot_4_stage_sankey(self, df, task_base, title_prefix):
    c_true_3m = task_base.replace('6m', '3m')
    c_pred_3m = f"Pred_{c_true_3m}"
    c_pred_6m = f"Pred_{task_base}"
    c_true_6m = task_base

    cols = [c_true_3m, c_pred_3m, c_pred_6m, c_true_6m]
    if not all(c in df.columns for c in cols): return

    valid = df[df[cols].isin([0, 1]).all(axis=1)]
    if len(valid) == 0: return

    # Nodes: Layer 0, 1, 2, 3 (8 nodes in total)
    labels = [
        "3m True (Normal)", "3m True (Abnormal)",
        "3m Pred (Normal)", "3m Pred (Abnormal)",
        "6m Pred (Normal)", "6m Pred (Abnormal)",
        "6m True (Normal)", "6m True (Abnormal)"
    ]
    node_colors = [self.COLOR_NORMAL, self.COLOR_ABNORMAL] * 4

    # Count links
    source, target, value = [], [], []
    for i in range(3):
        col_src = cols[i]
        col_tgt = cols[i + 1]
        counts = valid.groupby([col_src, col_tgt]).size().to_dict()
        for (sv, tv), cnt in counts.items():
            source.append(int(sv) + i * 2)
            target.append(int(tv) + (i + 1) * 2)
            value.append(cnt)

    link_colors = self.get_node_link_colors(source, target)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=labels, color=node_colors
        ),
        link=dict(source=source, target=target, value=value, color=link_colors)
    )])

    fig.update_layout(
        title_text=f"<b>Complete Clinical & Model Pathway: {title_prefix}</b><br><sup>(True 3m → Pred 3m → Pred 6m → True 6m)</sup>",
        font=dict(size=14, family="Times New Roman"), # Increase font size to adapt high-resolution
        width=1400, height=700,
        plot_bgcolor='white', paper_bgcolor='white'
    )

    # Export as high-resolution PNG (scale=6, simulate ~600 DPI)
    png_path = os.path.join(self.expr_path, f"Sankey_4Stage_Ultimate_{title_prefix}_600dpi.png")
    fig.write_image(png_path, scale=6)
    self.logger.info(f"  [+] Successfully generated: {png_path}")
