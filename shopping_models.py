"""
Modelado y visualización (sin terminal): comportamiento de compras

Salida:
- Imágenes (.png) y métricas (.csv/.txt) en la carpeta `resultados/`

Dependencias:
  pip install pandas numpy matplotlib seaborn scikit-learn
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# 1) Cargar datos
# =========================

DATA_PATH = Path("shopping_behavior_updated.csv")
RESULTS_DIR = Path("resultados")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 40

df = pd.read_csv(DATA_PATH)


# =========================
# 2) EDA: estadística descriptiva y distribución (IMÁGENES)
# =========================

sns.set_theme(style="whitegrid")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()


def _save_table_as_png(table_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.35 * len(table_df)))
    ax.axis("off")
    ax.set_title(title, pad=12)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


describe_num = df[numeric_cols].describe().round(3)
_save_table_as_png(
    describe_num,
    RESULTS_DIR / "01_descriptiva_numericas.png",
    "Estadística descriptiva (variables numéricas)",
)


# Distribuciones: histogramas + KDE (numéricas)
if numeric_cols:
    n = len(numeric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, bins=25, color="#3b82f6")
        ax.set_title(f"Distribución: {col}")
        ax.set_xlabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_distribuciones_numericas_hist_kde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Boxplots (numéricas)
if numeric_cols:
    n = len(numeric_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(x=df[col], ax=ax, color="#22c55e")
        ax.set_title(f"Boxplot: {col}")
        ax.set_xlabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_boxplots_numericas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Barras (categóricas seleccionadas y legibles)
cat_for_bars = [
    c
    for c in [
        "Gender",
        "Category",
        "Season",
        "Payment Method",
        "Frequency of Purchases",
        "Subscription Status",
        "Discount Applied",
        "Size",
    ]
    if c in df.columns
]

if cat_for_bars:
    n = len(cat_for_bars)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cat_for_bars):
        ax = axes[i]
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=ax, color="#a855f7")
        ax.set_title(f"Conteo: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
        ax.tick_params(axis="x", rotation=35)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_barras_categoricas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Correlación (solo numéricas)
if numeric_cols:
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_correlacion_numericas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 3) Regresión lineal: predecir Purchase Amount (USD)
# =========================

# Para que sea coherente como predicción, evitamos variables que normalmente se conocen DESPUÉS
# (por ejemplo, "Review Rating"). Usamos variables demográficas y de contexto de compra.
target_reg = "Purchase Amount (USD)"

features_reg = [
    c
    for c in [
        "Age",
        "Gender",
        "Item Purchased",
        "Category",
        "Location",
        "Size",
        "Color",
        "Season",
        "Subscription Status",
        "Previous Purchases",
        "Payment Method",
        "Frequency of Purchases",
    ]
    if c in df.columns
]

df_reg = df.dropna(subset=[target_reg])
X_reg = df_reg[features_reg]
y_reg = df_reg[target_reg]

cat_cols_reg = [c for c in features_reg if X_reg[c].dtype == "object"]
num_cols_reg = [c for c in features_reg if c not in cat_cols_reg]

preprocess_reg = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_reg),
        ("num", "passthrough", num_cols_reg),
    ],
    remainder="drop",
)

pipe_linreg = Pipeline(
    steps=[
        ("preprocess", preprocess_reg),
        ("model", LinearRegression()),
    ]
)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores_reg = cross_validate(
    pipe_linreg,
    X_reg,
    y_reg,
    cv=cv_reg,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    },
    n_jobs=None,
    return_train_score=False,
)

reg_summary = pd.DataFrame(
    {
        "modelo": ["LinearRegression"],
        "target": [target_reg],
        "cv_splits": [cv_reg.get_n_splits()],
        "r2_mean": [float(np.mean(scores_reg["test_r2"]))],
        "r2_std": [float(np.std(scores_reg["test_r2"]))],
        "rmse_mean": [float(-np.mean(scores_reg["test_rmse"]))],
        "rmse_std": [float(np.std(-scores_reg["test_rmse"]))],
        "mae_mean": [float(-np.mean(scores_reg["test_mae"]))],
        "mae_std": [float(np.std(-scores_reg["test_mae"]))],
    }
)
reg_summary.to_csv(RESULTS_DIR / "06_metricas_cv_regresion_lineal.csv", index=False)


# =========================
# 4) Regresión logística: predecir Discount Applied (Yes/No)
# =========================

target_clf = "Discount Applied"
df_clf = df.dropna(subset=[target_clf]).copy()
df_clf["discount_bin"] = (df_clf[target_clf] == "Yes").astype(int)

features_clf = [
    c
    for c in [
        "Age",
        "Gender",
        "Item Purchased",
        "Category",
        "Location",
        "Size",
        "Color",
        "Season",
        "Subscription Status",
        "Previous Purchases",
        "Payment Method",
        "Frequency of Purchases",
    ]
    if c in df_clf.columns
]

X_clf = df_clf[features_clf]
y_clf = df_clf["discount_bin"]

cat_cols_clf = [c for c in features_clf if X_clf[c].dtype == "object"]
num_cols_clf = [c for c in features_clf if c not in cat_cols_clf]

preprocess_clf = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_clf),
        ("num", "passthrough", num_cols_clf),
    ],
    remainder="drop",
)

pipe_logreg = Pipeline(
    steps=[
        ("preprocess", preprocess_clf),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores_log = cross_validate(
    pipe_logreg,
    X_clf,
    y_clf,
    cv=cv_clf,
    scoring={
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    },
    n_jobs=None,
    return_train_score=False,
)

log_summary = pd.DataFrame(
    {
        "modelo": ["LogisticRegression"],
        "target": [target_clf],
        "cv_splits": [cv_clf.get_n_splits()],
        "roc_auc_mean": [float(np.mean(scores_log["test_roc_auc"]))],
        "roc_auc_std": [float(np.std(scores_log["test_roc_auc"]))],
        "accuracy_mean": [float(np.mean(scores_log["test_accuracy"]))],
        "accuracy_std": [float(np.std(scores_log["test_accuracy"]))],
        "f1_mean": [float(np.mean(scores_log["test_f1"]))],
        "f1_std": [float(np.std(scores_log["test_f1"]))],
        "precision_mean": [float(np.mean(scores_log["test_precision"]))],
        "precision_std": [float(np.std(scores_log["test_precision"]))],
        "recall_mean": [float(np.mean(scores_log["test_recall"]))],
        "recall_std": [float(np.std(scores_log["test_recall"]))],
    }
)
log_summary.to_csv(RESULTS_DIR / "07_metricas_cv_regresion_logistica.csv", index=False)


# =========================
# 5) Modelo predictivo final (coherente con el caso): comparar y visualizar
# =========================

# Usamos el mismo objetivo de negocio (Discount Applied) y comparamos un modelo más flexible.
pipe_rf = Pipeline(
    steps=[
        ("preprocess", preprocess_clf),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)

scores_rf = cross_validate(
    pipe_rf,
    X_clf,
    y_clf,
    cv=cv_clf,
    scoring={
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    },
    n_jobs=None,
    return_train_score=False,
)

rf_summary = pd.DataFrame(
    {
        "modelo": ["RandomForestClassifier"],
        "target": [target_clf],
        "cv_splits": [cv_clf.get_n_splits()],
        "roc_auc_mean": [float(np.mean(scores_rf["test_roc_auc"]))],
        "roc_auc_std": [float(np.std(scores_rf["test_roc_auc"]))],
        "accuracy_mean": [float(np.mean(scores_rf["test_accuracy"]))],
        "accuracy_std": [float(np.std(scores_rf["test_accuracy"]))],
        "f1_mean": [float(np.mean(scores_rf["test_f1"]))],
        "f1_std": [float(np.std(scores_rf["test_f1"]))],
        "precision_mean": [float(np.mean(scores_rf["test_precision"]))],
        "precision_std": [float(np.std(scores_rf["test_precision"]))],
        "recall_mean": [float(np.mean(scores_rf["test_recall"]))],
        "recall_std": [float(np.std(scores_rf["test_recall"]))],
    }
)
rf_summary.to_csv(RESULTS_DIR / "08_metricas_cv_modelo_final_random_forest.csv", index=False)


# Visualizaciones del modelo final (sobre un holdout, solo para graficar ROC/CM)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_clf,
)

pipe_logreg.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_estimator(pipe_logreg, X_test, y_test, ax=ax, name="LogisticRegression")
RocCurveDisplay.from_estimator(pipe_rf, X_test, y_test, ax=ax, name="RandomForest")
ax.set_title("ROC (holdout) - Descuento aplicado")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "09_roc_descuento_holdout.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ConfusionMatrixDisplay.from_estimator(pipe_logreg, X_test, y_test, ax=axes[0], cmap="Blues")
axes[0].set_title("Matriz de confusión (LogisticRegression)")
ConfusionMatrixDisplay.from_estimator(pipe_rf, X_test, y_test, ax=axes[1], cmap="Greens")
axes[1].set_title("Matriz de confusión (RandomForest)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "10_matrices_confusion_holdout.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# Resumen breve en archivo (sin prints)
resumen_txt = "\n".join(
    [
        "Resultados generados en: resultados/",
        "",
        "EDA:",
        "- 01_descriptiva_numericas.png",
        "- 02_distribuciones_numericas_hist_kde.png",
        "- 03_boxplots_numericas.png",
        "- 04_barras_categoricas.png",
        "- 05_correlacion_numericas.png",
        "",
        "Modelos (métricas por validación cruzada):",
        "- 06_metricas_cv_regresion_lineal.csv",
        "- 07_metricas_cv_regresion_logistica.csv",
        "- 08_metricas_cv_modelo_final_random_forest.csv",
        "",
        "Visualización del modelo final (holdout):",
        "- 09_roc_descuento_holdout.png",
        "- 10_matrices_confusion_holdout.png",
        "",
        "Notas:",
        "- 'Review Rating' no se usó como feature para predicción porque normalmente ocurre después de la compra.",
    ]
)
(RESULTS_DIR / "RESUMEN_RESULTADOS.txt").write_text(resumen_txt, encoding="utf-8")

