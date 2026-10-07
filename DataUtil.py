import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_curve, auc

class DataSet:
  
  def __init__(self, config):
    self.config = config
    
    self.train = pd.read_csv(config['train_path'])
    self.val = pd.read_csv(config['val_path'])
    self.test = pd.read_csv(config['test_path'])
  
  def merge(self) -> pd.DataFrame:
    '''
        Merge train, val, test three datasets
    '''
    dfs = [self.train, self.val, self.test]
    df = pd.concat(dfs, ignore_index=True)
    return df

  def create_delta_feature(self, df, month):
    delta_cols = []
    if month == 1:
      pairs = [('Post1m_S', 'Pre_S', 'Delta_S_1m_Pre'),
             ('Post1m_C', 'Pre_C', 'Delta_C_1m_Pre'),
             ('Post1m_SE', 'Pre_SE', 'Delta_SE_1m_Pre'),
             ('Post1m_IOP', 'Pre_IOP', 'Delta_IOP_1m_Pre'),
             ('Post1m_HOA', 'Pre_HOA', 'Delta_HOA_1m_Pre')]
    elif month == 3:
      pairs = [
        ('Post3m_S', 'Post1m_S', 'Delta_S_3m_1m'),
        ('Post3m_C', 'Post1m_C', 'Delta_C_3m_1m'),
        ('Post3m_SE', 'Post1m_SE', 'Delta_SE_3m_1m'),
        ('Post3m_IOP', 'Post1m_IOP', 'Delta_IOP_3m_1m'),
        ('Post3m_HOA', 'Post1m_HOA', 'Delta_HOA_3m_1m'),
        ('Macular_Thickness_Mean_3m', 'Macular_Thickness_Mean_1m', 'Delta_Macular_Thickness_3m_1m'),
        ('Retinal_Flow_Mean_3m', 'Retinal_Flow_Mean_1m', 'Delta_Retinal_Flow_3m_1m')
      ]
    else:
      raise ValueError(f"month {month} is not supported")
    
    for post_col, pre_col, delta_col in pairs:
      if post_col in df.columns and pre_col in df.columns:
        df[delta_col] = df[post_col] - df[pre_col]
        delta_cols.append(delta_col)
        (f' -> Create delta feature successfully: {delta_col}')
    return df, delta_cols
  
class StatisticUtil:
  CLASS_NAMES = {2: ['Normal', 'Abnormal'], 3: ['Monofocal', 'EDOF', 'Multifocal']}
  def __init__(self, config):
    self.config = config
  
  def calculate_specificity(self, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        spec = tn / (tn + fp) if tn + fp > 0 else 0.0
        specificities.append(spec)
    return specificities

  def calculate_metrics_with_ci(self, y_true, y_pred, y_prob, n_bootstraps=1000, is_binary: bool = True):
      seed = self.config['seed']
      labels = [0, 1] if is_binary else [0, 1, 2]
      rng = np.random.RandomState(seed)
      stats = {
          'Accuracy': [], 'Macro_F1': [], 'Macro_Sensitivity': [], 'Macro_Specificity': [], 'AUC': []
      }

      for _ in range(n_bootstraps):
          indices = rng.randint(0, len(y_true), len(y_true))
          if len(np.unique(y_true[indices])) < len(labels): continue
          y_t, y_p = y_true[indices], y_pred[indices]
          if y_prob.ndim == 2: auc_value = roc_auc_score(y_t, y_prob[indices], multi_class='ovr', average='macro', labels=labels)
          else: auc_value = roc_auc_score(y_t, y_prob[indices])

          stats['Accuracy'].append(accuracy_score(y_t, y_p))
          stats['Macro_F1'].append(f1_score(y_t, y_p, labels=labels, average='macro', zero_division=0))
          stats['Macro_Sensitivity'].append(recall_score(y_t, y_p, labels=labels, average='macro', zero_division=0))
          stats['Macro_Specificity'].append(np.mean(self.calculate_specificity(y_t, y_p, labels)))
          stats['AUC'].append(auc_value)
      
      result = {}
      for k, v in stats.items():
          if len(v) == 0:
              result[k] = 'N/A'
              continue
          mean, lower, upper = np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)
          result[k] = f"{mean:.4f} ({lower:.4f} - {upper:.4f})"
      
      return result
    
  def calculate_class_metrics_with_ci(self,y_true, y_pred, y_prob, n_bootstraps=1000):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    num_classes = y_prob.shape[1]
    class_names = self.CLASS_NAMES[num_classes]
    metric_names = ['Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'PR_AUC']
    
    # BootStrap
    rng = np.random.RandomState(self.config['seed'])
    collected = {cname: {m: [] for m in metric_names} for cname in class_names}
    for _ in range(n_bootstraps):
      indices = rng.randint(0, len(y_true), len(y_true))
      y_t, y_p = y_true[indices], y_pred[indices]
      for c, cname in enumerate(class_names):
        if np.sum(y_t == c) == 0: continue    # If the class is empty, skip this round
        values = self.metrics_from_counts(*self.class_counts(y_t, y_p, c))
        for m in metric_names[:-1]:
          if values[m] is not None: collected[cname][m].append(values[m])
        pr = self.calculate_pr_auc((y_t == c).astype(int), y_prob[indices, c])
        if pr is not None: collected[cname]['PR_AUC'].append(pr)

    # --- Point estimate ---
    rows = []
    for c, cname in enumerate(class_names):
      tp, fp, fn, tn = self.class_counts(y_true, y_pred, c)
      values = self.metrics_from_counts(tp, fp, fn, tn)
      values['PR_AUC'] = self.calculate_pr_auc((y_true == c).astype(int), y_prob[:, c])
      row = {'Class': cname, 'Support': int(np.sum(y_true == c)),
             'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
      for m in metric_names:
        samples = collected[cname][m]
        row[m] = values[m]
        row[f'{m}_CI_lower'] = float(np.percentile(samples, 2.5)) if samples else None
        row[f'{m}_CI_upper'] = float(np.percentile(samples, 97.5)) if samples else None
        row[f'{m}_n_boot'] = len(samples)
      rows.append(row)
    return pd.DataFrame(rows)
    
  @staticmethod
  def class_counts(y_true, y_pred, label):
    '''
    OVR class count.
    '''
    is_true = (y_true == label)
    is_pred = (y_pred == label)
    tp = int(np.sum(is_true & is_pred))
    fp = int(np.sum(~is_true & is_pred))
    fn = int(np.sum(is_true & ~is_pred))
    tn = int(np.sum(~is_true & ~is_pred))
    return tp, fp, fn, tn
  
  @staticmethod
  def metrics_from_counts(tp, fp, fn, tn):
    return {
      'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else None,
      'Specificity': tn / (tn + fp) if (tn + fp) > 0 else None,
      'Precision':   tp / (tp + fp) if (tp + fp) > 0 else None,
      'NPV':         tn / (tn + fn) if (tn + fn) > 0 else None,
      'F1':          (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None,
    }
    
  @staticmethod
  def calculate_pr_auc(y_is_class, y_prob):
    if len(np.unique(y_is_class)) < 2: return None
    precision, recall, _ = precision_recall_curve(y_is_class, y_prob)
    return float(auc(recall, precision))
    
