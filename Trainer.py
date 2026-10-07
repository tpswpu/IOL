import time
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import DataUtil as du
import DrawUtil as dru

class Trainer:
  def __init__(self, config, model, cascade_mode, logger, time_stamp):
    self.config = config
    self.model = model
    self.cascade_mode = cascade_mode
    self.logger = logger
    self.time_stamp = time_stamp
    
    expr_path = os.path.join(config["output_path"], f"{time_stamp}")
    self.expr_path = os.path.join(expr_path, f"exp_{model}_{cascade_mode}")
    os.makedirs(self.expr_path, exist_ok=True)
    
    self.data = du.DataSet(config).merge()
    self.su = du.StatisticUtil(config)
    self.dru = dru.DrawUtil(config, self.expr_path, logger)
  
  def fit_xgb(self, X_tr, y_tr, num_classes, seed, fold=None):
    params = self.config['model_params']['xgb'].copy()
    params['random_state'] = seed
    params['n_jobs'] = -1
    params['objective'] = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
    params['eval_metric'] = 'mlogloss' if num_classes > 2 else 'logloss'
    model = xgb.XGBClassifier(**params)
    weights = compute_sample_weight(class_weight='balanced', y=y_tr)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_tr, inner_es = next(inner.split(X_tr, y_tr))
    model.fit(X_tr[inner_tr], y_tr[inner_tr], sample_weight=weights[inner_tr], eval_set=[(X_tr[inner_es], y_tr[inner_es])], verbose=False)
    if fold is not None: self.logger.info(f"  [fold {fold}] best_iteration = {model.best_iteration}")
    return model

  def nested_cascade(self, cascade_spec, train_keys, val_keys, fold):
    """Regenerate cascade features per outer fold so that the validation fold
    never contributes to the Stage-2 model that produces them."""
    casc = {}
    for col, src in cascade_spec.items():
      train_vals = np.full(len(train_keys), np.nan)
      val_vals = np.full(len(val_keys), np.nan)
      src_pos = {key: i for i, key in enumerate(src['idx'])}
      tr_rows = [(r, src_pos[k]) for r, k in enumerate(train_keys) if k in src_pos]
      va_rows = [(r, src_pos[k]) for r, k in enumerate(val_keys) if k in src_pos]
      if len(tr_rows) >= 10:
        pos_tr = [r for r, _ in tr_rows]
        idx_tr = [j for _, j in tr_rows]
        X_src, y_src = src['X'][idx_tr], src['y'][idx_tr]
        if len(np.unique(y_src)) > 1 and np.min(np.bincount(y_src)) >= 5:
          inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['seed'] + fold)
          inner_oof = np.zeros(len(y_src))
          for g, (itr, iva) in enumerate(inner.split(X_src, y_src)):
            sub = self.fit_xgb(X_src[itr], y_src[itr], src['num_classes'], self.config['seed'] + fold * 10 + g)
            inner_oof[iva] = sub.predict_proba(X_src[iva])[:, 1]
          train_vals[pos_tr] = inner_oof
          full = self.fit_xgb(X_src, y_src, src['num_classes'], self.config['seed'] + fold * 10 + 9)
          if va_rows: val_vals[[r for r, _ in va_rows]] = full.predict_proba(src['X'][[j for _, j in va_rows]])[:, 1]
      casc[col] = (train_vals, val_vals)
    return casc

  def assemble_matrix(self, valid_df, feature_cols, base_cols, cascade_cols, rows, casc, side):
    block = valid_df.iloc[rows][base_cols].copy()
    for col in cascade_cols: block[col] = casc[col][0 if side == 'train' else 1]
    return block[feature_cols].values

  def train_stage(self, df, task_name, num_classes, feature_cols, fig_report_list, is_binary_stage2: bool = False, cascade_spec=None):
    self.logger.info(f"\n{'=' * 60}\n>>> Running 5-Fold CV: {task_name}\n{'=' * 60}")
    
    valid_df = df[df[task_name] != -1].copy()
    if len(valid_df) < 10:
        self.logger.warning(f"  [Skip] {task_name} Sample size is too small.")
        return df, [], []

    X = valid_df[feature_cols].values
    y = valid_df[task_name].astype(int).values

    # Save OOF Predictions (Out-of-Fold)
    oof_preds = np.zeros(len(valid_df))
    oof_probs = np.zeros(len(valid_df)) if num_classes == 2 else np.zeros((len(valid_df), num_classes))

    oof_prob_matrix = np.zeros((len(valid_df), num_classes), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['seed'])

    model_params = self.config['model_params'][self.model]
    fold_aucs = []

    cascade_cols = list(cascade_spec) if cascade_spec else []
    base_cols = [c for c in feature_cols if c not in cascade_cols]
    stage_keys = valid_df.index.tolist()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
      weights = compute_sample_weight(class_weight='balanced', y=y[train_idx])
      if cascade_spec:
        casc = self.nested_cascade(cascade_spec, [stage_keys[i] for i in train_idx], [stage_keys[i] for i in val_idx], fold)
        X_tr = self.assemble_matrix(valid_df, feature_cols, base_cols, cascade_cols, train_idx, casc, 'train')
        X_va = self.assemble_matrix(valid_df, feature_cols, base_cols, cascade_cols, val_idx, casc, 'val')
      else:
        X_tr, X_va = X[train_idx], X[val_idx]
      y_tr, y_va = y[train_idx], y[val_idx]

      if self.model == 'xgb':
        model = self.fit_xgb(X_tr, y_tr, num_classes, self.config['seed'] + fold, fold)
      elif self.model == 'dummy':
        model = DummyClassifier(**model_params)
        model.fit(X_tr, y_tr)
      elif self.model in ('lr', 'rf', 'svm'):
        constructors = {
            'lr': LogisticRegression,
            'rf': RandomForestClassifier,
            'svm': SVC,
        }
        params = model_params.copy()
        params['random_state'] = self.config['seed'] + fold
        estimator = constructors[self.model](**params)
        steps = [('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]

        # Logistic and SVM require feature scaling.
        if self.model in ('lr', 'svm'): steps.append(('scaler', StandardScaler()))

        steps.append(('clf', estimator))
        model = Pipeline(steps)

        model.fit(X_tr, y_tr, clf__sample_weight=weights)
        
      else: raise ValueError(f'Model is not supported: {self.model}')

      probs_all = model.predict_proba(X_va)
      oof_prob_matrix[val_idx] = probs_all
      # OOF Prediction
      if num_classes == 2:
          probs = probs_all[:, 1]
          oof_probs[val_idx] = probs
          oof_preds[val_idx] = (probs >= 0.5).astype(int)
          if len(np.unique(y_va)) == 2:fold_aucs.append(roc_auc_score(y_va, probs))
      else:
          oof_probs[val_idx] = probs_all
          oof_preds[val_idx] = np.argmax(probs_all, axis=1)
          
    self.logger.info(f"  -> 5-Fold CV Completed. OOF Samples: {len(y)}")
    if fold_aucs: self.logger.info(f"  -> Fold AUC: {[f'{a:.3f}' for a in fold_aucs]} | Mean: {np.mean(fold_aucs):.3f}")
    
    # --- Global Statistics and Charts ---
    metrics = self.su.calculate_metrics_with_ci(y, oof_preds, oof_probs, self.config.get('n_bootstrap', 1000), num_classes==2)
    report_lines = [f"\nTask: {task_name} (OOF Valid Samples: {len(y)})"]
    for k, v in metrics.items(): report_lines.append(f"  {k}: {v}")
    
    class_df = self.su.calculate_class_metrics_with_ci(y, oof_preds, oof_prob_matrix, self.config.get('n_bootstrap', 1000))
    metric_names = ['Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'PR_AUC']

    report_lines.append("  --- Class-specific metrics (OvR, point estimate with 95% CI) ---")
    for _, r in class_df.iterrows():
        cells = []
        for m in metric_names:
            if pd.isna(r[m]): cells.append(f"{m}=NA")
            elif pd.isna(r[f'{m}_CI_lower']): cells.append(f"{m}={r[m]:.4f}")
            else: cells.append(f"{m}={r[m]:.4f} ({r[f'{m}_CI_lower']:.4f} - {r[f'{m}_CI_upper']:.4f})")
        report_lines.append(f"  [{r['Class']}] n={int(r['Support'])} | " + " | ".join(cells))

    self.dru.confusion_matrix(task_name, y, oof_preds, fig_report_list, num_classes==2)
    self.dru.roc_curve(task_name, y, oof_probs, fig_report_list, num_classes==2)
    if self.model == 'xgb':
      weights_all = compute_sample_weight(class_weight='balanced', y=y)
      final_params = model_params.copy()
      final_params['random_state'] = self.config['seed']
      final_params['n_jobs'] = -1
      final_params['objective'] = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
      final_params['eval_metric'] = 'mlogloss' if num_classes > 2 else 'logloss'
      final_model = xgb.XGBClassifier(**final_params)
      final_model.fit(X, y, sample_weight=weights_all, eval_set=[(X, y)], verbose=False)
      final_model.save_model(os.path.join(self.expr_path, f"model_{task_name}_global.json"))
      self.dru.shap_values(task_name, final_model, X, feature_cols, fig_report_list, num_classes==2)

    appended_cols = []
    cascade_source = None
    if self.model == 'xgb' and num_classes == 2 and is_binary_stage2:
      if self.cascade_mode == 'soft':
        new_col = f"Prob_{task_name}"
        df[new_col] = np.nan
        df.loc[valid_df.index, new_col] = oof_probs
        appended_cols.append(new_col)
        self.logger.info(f"  -> OOF probability of {task_name} has been injected into the main table for cascading.")
      elif self.cascade_mode == 'hard':
        new_col = f"Pred_{task_name}"
        df[new_col] = np.nan
        df.loc[valid_df.index, new_col] = oof_preds
        appended_cols.append(new_col)
        self.logger.info(f"===>Hard Cascade Injection: {new_col}")
      elif self.cascade_mode == "none":
        self.logger.info("===>No Cascade Injection")
      if appended_cols:
        cascade_source = {'col': appended_cols[0],'idx': valid_df.index.tolist(),'X': valid_df[feature_cols].values,'y': y,'num_classes': num_classes}

    return df, report_lines, appended_cols, cascade_source
  
  def train(self):
    self.logger.info(f"\n\n{'*' * 70}")
    self.logger.info(f"🚀 Experiment Mode: CASCADE MODE = '{self.cascade_mode.upper()}'")
    self.logger.info(f"{'*' * 70}\n")
    
    with open(os.path.join(self.expr_path, "config_backup.yaml"), "w") as f:
      yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
      
    df = self.data.copy()
    delta_cols_1m, delta_cols_3m = [], []
    
    df, delta_cols_1m = du.DataSet(self.config).create_delta_feature(df, 1)
    df, delta_cols_3m = du.DataSet(self.config).create_delta_feature(df, 3)
    
    # Feature selection
    feat_forIOL_pre = [c for c in self.config.get('features_forIOL_pre_op', []) if c in df.columns]

    feat_forMtf_pre = [c for c in self.config.get('features_forMtf_pre_op', []) if c in df.columns]
    feat_forMtf_post_1m = [c for c in self.config.get('features_forMtf_post_1m', []) if c in df.columns]
    feat_forMtf_post_3m = [c for c in self.config.get('features_forMtf_post_3m', []) if c in df.columns]
    feat_forMtf_post_6m = [c for c in self.config.get('features_forMtf_post_6m', []) if c in df.columns]

    feat_forRetinal_pre = [c for c in self.config.get('features_forRetinal_pre_op', []) if c in df.columns]
    feat_forRetinal_post_1m = [c for c in self.config.get('features_forRetinal_post_1m', []) if c in df.columns]
    feat_forRetinal_post_3m = [c for c in self.config.get('features_forRetinal_post_3m', []) if c in df.columns]
    feat_forRetinal_post_6m = [c for c in self.config.get('features_forRetinal_post_6m', []) if c in df.columns]
    
    final_report = [f"=== Ultimate 5-Fold CV Cascaded Report ===", f"Total Samples: {len(df)}", "=" * 50]
    fig_report_list = ["# Figure Legend Report\n\n*This report is intended to provide a standardized description of the figures generated in the paper.\n","=" * 60 + "\n"]

    tasks = self.config.get('tasks')
    # --- STAGE 1: Pre-Operative ---
    stage1_feats = [c for c in feat_forIOL_pre if c != 'IOL']
    for task_name, num_classes in tasks.get('tasks_stage1_iol', {}).items():
        df, rpt, _, _ = self.train_stage(df, task_name, num_classes, stage1_feats, fig_report_list)
        final_report.extend(rpt)

    # --- STAGE 2: 3-month (Output Soft Probability Features) ---
    stage3_forMtf_cascades = {}
    stage3_forRetinal_cascades = {}
    for task_name, num_classes in tasks.get('tasks_stage2_3month', {}).items():
        if 'MTF' in task_name:
            stage2_forMtf_feats = feat_forMtf_pre + feat_forMtf_post_1m + feat_forMtf_post_3m + delta_cols_1m
            df, rpt, _, cascade_source = self.train_stage(df, task_name, num_classes, stage2_forMtf_feats, fig_report_list,is_binary_stage2=True)
            final_report.extend(rpt)
            if cascade_source: stage3_forMtf_cascades[cascade_source['col']] = cascade_source
        elif 'erg' in task_name:
            stage2_forRetinal_feats = feat_forRetinal_pre + feat_forRetinal_post_1m + feat_forRetinal_post_3m
            df, rpt, _, cascade_source = self.train_stage(df, task_name, num_classes, stage2_forRetinal_feats, fig_report_list,is_binary_stage2=True)
            final_report.extend(rpt)
            if cascade_source: stage3_forRetinal_cascades[cascade_source['col']] = cascade_source


    # --- STAGE 3: 6-month (Ultimate Trajectory Prediction) ---
    for task_name, num_classes in tasks.get('tasks_stage3_6month', {}).items():
        if 'MTF' in task_name:
            # 6-month task separately constructs input: keep original features, but only add delta_3m to
            # avoid inheriting Stage 2 delta_1m.
            stage3_forMtf_feats = (feat_forMtf_pre + feat_forMtf_post_1m + feat_forMtf_post_3m + feat_forMtf_post_6m + list(stage3_forMtf_cascades) + delta_cols_3m[:-2])
            df, rpt, _, _ = self.train_stage(df, task_name, num_classes, stage3_forMtf_feats, fig_report_list, cascade_spec=stage3_forMtf_cascades)
            final_report.extend(rpt)
        elif 'erg' in task_name:
            selected_delta_cols_3m = [ col for col in ["Delta_Macular_Thickness_3m_1m", "Delta_Retinal_Flow_3m_1m"] if col in delta_cols_3m ]
            stage3_forRetinal_feats = ( feat_forRetinal_pre + feat_forRetinal_post_1m + feat_forRetinal_post_3m + feat_forRetinal_post_6m + list(stage3_forRetinal_cascades) + selected_delta_cols_3m )
            df, rpt, _, _ = self.train_stage(df, task_name, num_classes, stage3_forRetinal_feats, fig_report_list, cascade_spec=stage3_forRetinal_cascades)
            final_report.extend(rpt)

            if self.model == "xgb" and self.cascade_mode == "hard":
              self.dru.plot_2_stage_sankey(df, task_name, task_name)
              self.dru.plot_4_stage_sankey(df, task_name, task_name)

    #  Save evaluation metrics report for the final stage
    with open(os.path.join(self.expr_path, "results_report_OOF_CV.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(final_report))

    #  Save figure legend report for the final stage
    with open(os.path.join(self.expr_path, "figure_report.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(fig_report_list))

    self.logger.info(f"\n🏆 Cascaded 5-Fold CV Validation Completed!")
    self.logger.info(f"  --> Evaluation metrics report saved at: {os.path.join(self.expr_path, 'results_report_OOF_CV.txt')}")
    self.logger.info(f"  --> Figure legend report saved at: {os.path.join(self.expr_path, 'figure_report.txt')}")
