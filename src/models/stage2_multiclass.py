"""Módulo do Estágio 2: Classificador multiclasse para tipificação de fraudes (ADRs 0001 e 0003)."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


@dataclass
class Stage2Results:
    """Resultados de treino e avaliação do Estágio 2 (Tipificação Multiclasse)."""

    model: lgb.LGBMClassifier
    classes: list[str]
    val_macro_f1: float
    val_balanced_acc: float
    confusion_matrix: np.ndarray


def train_stage2_multiclass(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> Stage2Results:
    """Treina o classificador LightGBM multiclasse ponderado para discriminar

    entre as 4 modalidades de fraude em instâncias fraudulentas.
    """
    classes = sorted(y_train.unique())

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        class_weight="balanced",
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)

    macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_val, y_val_pred))
    cm = confusion_matrix(y_val, y_val_pred, labels=classes)

    return Stage2Results(
        model=clf,
        classes=classes,
        val_macro_f1=macro_f1,
        val_balanced_acc=balanced_acc,
        confusion_matrix=cm,
    )
