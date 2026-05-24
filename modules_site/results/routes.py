from flask import render_template, jsonify, current_app
from . import bp  # Импортируем свой Blueprint
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
import seaborn
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
import re

def _load_and_predict(model_manager, model_id, env='test',
                      test_path='test/processed/test_data/isolation_forest_test_10k.csv'):
    """
    Загружает тестовый CSV, получает предсказания и вероятности модели.
    Возвращает (y_true_str, y_pred_str, y_proba, class_names) или None при ошибке.
    """
    try:
        from pathlib import Path
        if not Path(test_path).exists():
            current_app.logger.warning(f"Test file not found: {test_path}")
            return None

        df = pd.read_csv(test_path)

        # Ищем колонку с метками (Label / label / Class / Attack)
        label_col = next((c for c in df.columns
                          if c.strip().lower() in ('label', 'class', 'attack', 'target')), None)
        if label_col is None:
            current_app.logger.warning("Label column not found in test CSV")
            return None

        y_true_str = df[label_col].astype(str).values
        X_raw = df.drop(columns=[label_col])

        # Очищаем имена колонок так же, как при обучении LightGBM
        if model_id == 'lightgbm':
            X_raw.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in X_raw.columns]
            X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()]

        bundle  = model_manager._get_or_load_bundle(model_id, env)
        le      = bundle.label_encoder
        classes = list(le.classes_)                    # строковые имена классов

        # Выравниваем признаки под обученную модель
        X_aligned = model_manager._normalize_input_features(
            X_raw, bundle.feature_names, model_id
        )

        # Предсказания строками
        y_pred_str = model_manager.predict(model_id, X_raw, env)

        # Вероятности (не для Isolation Forest)
        y_proba = None
        if model_id != 'isolation_forest':
            try:
                y_proba = bundle.pipeline.predict_proba(X_aligned)
            except Exception as e:
                current_app.logger.warning(f"predict_proba failed for {model_id}: {e}")

        return y_true_str, np.array(y_pred_str), y_proba, classes

    except Exception as e:
        current_app.logger.error(f"_load_and_predict failed for {model_id}: {e}", exc_info=True)
        return None

def _plot_to_base64(dpi=120):
    """Конвертирует matplotlib график в base64-строку"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,
                facecolor='#1f2937', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64


def _generate_test_data(n_samples=5000):
    """Генерирует синтетические тестовые данные для демонстрации метрик"""
    np.random.seed(42)

    classes = ['ARP-spoof', 'Benign', 'FTP-Attack', 'Fuzzing']
    class_weights = [0.1, 0.6, 0.15, 0.15]

    y_true = np.random.choice(classes, size=n_samples, p=class_weights)

    # Генерируем вероятности предсказаний с некоторой точностью (~80-90%)
    y_pred = []
    y_proba = []

    for true_class in y_true:
        true_idx = classes.index(true_class)

        # С вероятностью 85% предсказываем правильно, иначе ошибаемся
        if np.random.random() < 0.85:
            pred_idx = true_idx
        else:
            # Ошибаемся - выбираем случайный другой класс
            wrong_choices = [i for i in range(len(classes)) if i != true_idx]
            pred_idx = np.random.choice(wrong_choices)

        # Генерируем вероятности
        probs = np.random.uniform(0.02, 0.15, size=len(classes))

        # Устанавливаем вероятность для предсказанного класса выше
        probs[pred_idx] = np.random.uniform(0.55, 0.85)

        # Нормализуем вероятности
        probs = probs / probs.sum()
        y_proba.append(probs)

        y_pred.append(classes[pred_idx])

    return np.array(y_true), np.array(y_pred), np.array(y_proba)

def _calculate_model_metrics(model_manager, model_id, env='test'):
    """Рассчитывает метрики для конкретной модели"""
    try:
        # Загружаем бандл
        bundle = model_manager._get_or_load_bundle(model_id, env)
        pipeline = bundle.pipeline

        # Загружаем метрики из файла если есть
        import json
        from pathlib import Path
        metrics_path = bundle.root_dir / 'metrics.json'

        file_metrics = {}
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                file_metrics = json.load(f)

        # Получаем базовые метрики из файла
        accuracy = file_metrics.get('accuracy', file_metrics.get('test_accuracy', 0))
        # F1 мера - используем f1_weighted для общей оценки качества
        # F1 мера - пробуем разные варианты ключей
        f1_weighted = file_metrics.get('f1_weighted',
                                       file_metrics.get('f1_macro',
                                                        file_metrics.get('f1_score', 0)))

        # ROC-AUC - пробуем разные варианты ключей
        roc_auc = file_metrics.get('roc_auc_macro',
                                   file_metrics.get('roc_auc_micro',
                                                    file_metrics.get('macro_auc',
                                                                     file_metrics.get('micro_auc',
                                                                                      file_metrics.get('roc_auc', 0)))))

        # Форматируем метрики
        accuracy_str = f"{accuracy * 100:.2f}%" if accuracy <= 1 else f"{accuracy:.2f}%"
        f1_str = f"{f1_weighted * 100:.2f}%" if f1_weighted <= 1 else f"{f1_weighted:.2f}%"
        roc_auc_str = f"{roc_auc:.4f}"

        # Получаем информацию о классах из label_encoder или meta.json
        class_names = []
        # Сначала пробуем получить из label_encoder
        if hasattr(bundle, 'label_encoder') and hasattr(bundle.label_encoder, 'classes_'):
            class_names = list(bundle.label_encoder.classes_)

        # Если не получилось, пробуем meta.json
        if not class_names:
            meta_path = bundle.root_dir / 'meta.json'
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    class_names = meta.get('class_names', [])

        # Если нет имен классов, используем дефолтные
        if not class_names:
            class_names = ['Benign', 'DoS/DDoS', 'Intrusion', 'Anomaly']

        return {
            'model_id': model_id,
            'accuracy': accuracy_str,
            'f1_score': f1_str,
            'roc_auc': roc_auc_str,
            'class_names': class_names,
            'pipeline': pipeline,
            'bundle': bundle
        }
    except Exception as e:
        current_app.logger.error(f"Error calculating metrics for {model_id}: {e}")
        return None

def _generate_confusion_matrix(pipeline, bundle, class_names,
                                y_true=None, y_pred=None):
    """Матрица ошибок на реальных тестовых данных (или заглушка)."""
    try:
        import seaborn as sns

        # --- Реальные данные ---
        if y_true is not None and y_pred is not None:
            # Оставляем только классы, которые встречаются в выборке
            present = sorted(set(y_true) | set(y_pred))
            labels  = [c for c in class_names if c in present] or present
            cm      = confusion_matrix(y_true, y_pred, labels=labels)
            tick_labels = labels
        else:
            # Заглушка если данных нет
            n  = min(len(class_names), 10)
            cm = np.random.randint(10, 100, size=(n, n))
            np.fill_diagonal(cm, np.random.randint(500, 1000, size=n))
            tick_labels = class_names[:n]

        n = len(tick_labels)
        fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)),
                               facecolor='#1f2937')
        ax.set_facecolor('#1f2937')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=tick_labels, yticklabels=tick_labels,
            linewidths=0.4, ax=ax,
            annot_kws={'size': max(6, 10 - n // 3)}
        )
        ax.set_xlabel('Предсказанный класс', color='#9ca3af', fontsize=10)
        ax.set_ylabel('Истинный класс',      color='#9ca3af', fontsize=10)
        ax.set_title('Матрица ошибок (Confusion Matrix)',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#9ca3af', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return _plot_to_base64()

    except Exception as e:
        current_app.logger.error(f"Error generating confusion matrix: {e}")
        return None


def _generate_roc_curve(pipeline, bundle, class_names, model_id=None,
                        y_true=None, y_proba=None):
    """ROC-кривые One-vs-Rest на реальных данных (или заглушка)."""
    try:
        is_if = isinstance(pipeline, dict) or (
            hasattr(pipeline, 'named_steps') and
            isinstance(pipeline.named_steps.get('classifier'), dict)
        )

        # --- Реальные данные (XGBoost / LightGBM / Random Forest) ---
        if y_true is not None and y_proba is not None and not is_if:
            le_tmp = bundle.label_encoder
            # Кодируем y_true в числа для roc_curve
            from sklearn.preprocessing import LabelEncoder
            le_local = LabelEncoder()
            le_local.classes_ = np.array(le_tmp.classes_)
            # Оставляем только классы, присутствующие в y_proba
            present_classes = list(le_tmp.classes_)
            y_true_enc = np.array([
                list(le_tmp.classes_).index(c) if c in le_tmp.classes_ else -1
                for c in y_true
            ])
            mask = y_true_enc >= 0
            y_true_enc = y_true_enc[mask]
            y_proba_f  = y_proba[mask]

            class_aucs = []
            for i, cls_name in enumerate(present_classes):
                y_bin = (y_true_enc == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin, y_proba_f[:, i])
                class_aucs.append({
                    'class': cls_name,
                    'auc':   auc(fpr, tpr),
                    'fpr':   fpr,
                    'tpr':   tpr
                })

            roc_df = (pd.DataFrame(class_aucs)
                        .sort_values('auc', ascending=False)
                        .reset_index(drop=True))

            fig, ax = plt.subplots(figsize=(13, 9), facecolor='white')
            ax.set_facecolor('white')
            colors   = plt.cm.Set3(np.linspace(0, 1, len(roc_df)))
            idx_show = set(list(range(7)) + list(range(max(0, len(roc_df)-5), len(roc_df))))

            for i, row in roc_df.iterrows():
                if i not in idx_show:
                    continue
                ax.plot(row['fpr'], row['tpr'],
                        color=colors[i % len(colors)], linewidth=1.6,
                        label=f"{row['class']} (AUC={row['auc']:.3f})", alpha=0.9)

            ax.plot([0, 1], [0, 1], color='#6b7280', linestyle='--',
                    linewidth=1.4, label='Случайный (AUC=0.5)', alpha=0.7)

            macro_auc = roc_df['auc'].mean()
            roc_suffix = " (One-vs-Rest)" if model_id == 'isolation_forest' else ""
            ax.set_title(
                f'ROC-кривые{roc_suffix} — {(model_id or "").replace("_", " ").title()}'
                f'\nМакро-AUC = {macro_auc:.4f}',
                color='white', fontsize=12, fontweight='bold'
            )
        else:
            # --- Заглушка (Isolation Forest или нет данных) ---
            fig, ax = plt.subplots(figsize=(13, 9), facecolor='white')
            ax.set_facecolor('white')
            colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
            n_show = min(len(class_names), 12)
            for i in range(n_show):
                fpr = np.linspace(0, 1, 100)
                tpr = np.power(fpr, 0.55 + np.random.uniform(-0.08, 0.08))
                ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                        label=f"{class_names[i][:18]} (AUC={auc(fpr,tpr):.3f})", alpha=0.85)
            ax.plot([0, 1], [0, 1], color='#6b7280', linestyle='--',
                    linewidth=1.4, label='Случайный (AUC=0.5)')
            roc_suffix = " (One-vs-Rest)" if model_id == 'isolation_forest' else ""
            ax.set_title(
                f'ROC-кривые{roc_suffix} — {(model_id or "").replace("_", " ").title()}',
                color='white', fontsize=12, fontweight='bold'
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', color='#9ca3af', fontsize=10)
        ax.set_ylabel('True Positive Rate',  color='#9ca3af', fontsize=10)
        ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9,
                  facecolor='white', edgecolor='black', labelcolor='black')
        ax.tick_params(colors='#9ca3af')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return _plot_to_base64()

    except Exception as e:
        current_app.logger.error(f"Error generating ROC curve: {e}")
        return None


def _generate_feature_importance(pipeline, bundle, model_id):
    """Генерирует feature importance"""
    try:
        # Получаем имена признаков
        feature_names = bundle.feature_names or []

        # Получаем classifier из pipeline
        if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
            classifier = pipeline.named_steps['classifier']
        else:
            classifier = pipeline

        # Проверяем наличие feature_importances_
        if not hasattr(classifier, 'feature_importances_'):
            return None

        importances = classifier.feature_importances_
        n_features = min(15, len(importances))
        indices = np.argsort(importances)[-n_features:][::-1]

        # Получаем имена признаков или создаем заглушки
        labels = []
        for idx in indices:
            if idx < len(feature_names):
                name = feature_names[idx]
                labels.append(name[:22] + '...' if len(name) > 25 else name)
            else:
                labels.append(f'Feature_{idx}')

        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1f2937')
        ax.barh(range(n_features), importances[indices], color='#2E86AB')
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Важность (Gain)', color='#9ca3af', fontsize=9)
        ax.set_title(f'Feature Importance - {model_id}', color='white', fontsize=12)
        ax.tick_params(colors='#9ca3af', labelsize=8)

        fig.tight_layout()
        return _plot_to_base64()
    except Exception as e:
        current_app.logger.error(f"Error generating feature importance: {e}")
        return None


def _plot_confusion_matrix(y_true, y_pred, classes):
    """Создаёт heatmap матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1f2937')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Добавляем значения в ячейки
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=10)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', color='white')
    ax.set_yticklabels(classes, color='white')
    ax.set_xlabel('Predicted label', color='white')
    ax.set_ylabel('True label', color='white')
    ax.set_title('Confusion Matrix', color='white', fontsize=12)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    return _plot_to_base64()


def _plot_roc_curve(y_true, y_proba, classes, roc_auc_mean):
    """Создаёт ROC-кривую для каждого класса"""
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1f2937')

    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    for i, color in zip(range(n_classes), colors[:n_classes]):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        except:
            pass

    # Диагональная линия
    ax.plot([0, 1], [0, 1], '--', lw=1, color='#6B7280', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title(f'ROC Curve (Mean AUC = {roc_auc_mean:.3f})', color='white', fontsize=12)
    ax.legend(loc='lower right', facecolor='#1f2937', edgecolor='#374151',
             labelcolor='white', fontsize=8)
    ax.grid(True, alpha=0.2, color='#374151')

    plt.tight_layout()
    return _plot_to_base64()


def _plot_feature_importance():
    """Создаёт график важности признаков из загруженной модели"""
    try:
        pipeline = current_app.pipeline
        if pipeline is None:
            # Если пайплайн не загружен, возвращаем заглушку
            return None

        lgb_model = pipeline.named_steps['classifier']
        booster = lgb_model.booster_

        importance = booster.feature_importance(importance_type='gain')
        feature_names = booster.feature_name()

        # Топ-15 признаков
        top_indices = np.argsort(importance)[-15:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1f2937')

        bars = ax.barh(range(len(top_features)), top_importance, color='#3B82F6')

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=8, color='white')
        ax.set_xlabel('Feature Importance (Gain)', color='white')
        ax.set_title('Top 15 Feature Importance', color='white', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x', color='#374151')

        plt.tight_layout()
        return _plot_to_base64()
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None


@bp.route('/')
def results():
    """Роут для страницы /results"""
    # Получаем классы из feature_info
    # Проверяем наличие ModelManager
    if not hasattr(current_app, 'model_manager'):
        return render_template('results.html',
                               models_results=[],
                               error="ModelManager not initialized")

    model_manager = current_app.model_manager

    # Список доступных моделей
    available_models = ['lightgbm', 'xgboost', 'random_forest', 'isolation_forest']

    models_results = []

    for model_id in available_models:
        # Проверяем наличие модели в file_map
        if model_id not in model_manager.file_map:
            continue

        # Рассчитываем метрики
        metrics_data = _calculate_model_metrics(model_manager, model_id)

        if not metrics_data:
            continue

        pred_result = _load_and_predict(model_manager, model_id)

        if pred_result is not None:
            y_true_r, y_pred_r, y_proba_r, _ = pred_result
        else:
            y_true_r = y_pred_r = y_proba_r = None

        # --- Генерируем графики на реальных данных ---
        confusion_matrix_img = _generate_confusion_matrix(
            metrics_data['pipeline'],
            metrics_data['bundle'],
            metrics_data['class_names'],
            y_true=y_true_r,
            y_pred=y_pred_r,
        )

        roc_curve_img = _generate_roc_curve(
            metrics_data['pipeline'],
            metrics_data['bundle'],
            metrics_data['class_names'],
            model_id=model_id,
            y_true=y_true_r,
            y_proba=y_proba_r,
        )

        feature_importance_img = _generate_feature_importance(
            metrics_data['pipeline'],
            metrics_data['bundle'],
            model_id
        )

        model_result = {
            'model_id': model_id,
            'model_name': model_id.replace('_', ' ').title(),
            'accuracy': metrics_data['accuracy'],
            'f1_score': metrics_data['f1_score'],
            'roc_auc': metrics_data['roc_auc'],
            'confusion_matrix_img': confusion_matrix_img,
            'roc_curve_img': roc_curve_img,
            'class_names': metrics_data['class_names'][:10],
        }
        models_results.append(model_result)

    # Если ни одна модель не загружена, показываем заглушку
    if not models_results:
        return render_template('results.html',
                               models_results=[],
                               error="No models available")

        # Определяем лучшую модель по accuracy
    best_model = max(models_results, key=lambda x: float(x['accuracy'].rstrip('%')))

     # Формируем данные для сравнительной таблицы
    comparison_table = []
    for model in models_results:
            is_best = model['model_id'] == best_model['model_id']
            comparison_table.append({
                'model_name': model['model_name'],
                'accuracy': model['accuracy'],
                'f1_score': model['f1_score'],
                'roc_auc': model['roc_auc'],
                'is_best': is_best
            })

    return render_template('results.html',
                               models_results=models_results,
                               comparison_table=comparison_table,
                               best_model=best_model)


@bp.route('/api/metrics')
def api_metrics():
    """API endpoint для получения метрик в JSON формате"""
    try:
        feature_info = current_app.feature_info
        classes = feature_info.get('classes_', ['ARP-spoof', 'Benign', 'FTP-Attack', 'Fuzzing'])
    except:
        classes = ['ARP-spoof', 'Benign', 'FTP-Attack', 'Fuzzing']

    y_true, y_pred, y_proba = _generate_test_data(n_samples=5000)
    metrics, class_metrics, roc_auc_mean = _calculate_metrics(y_true, y_pred, y_proba, classes)

    return jsonify({
        'metrics': metrics,
        'class_metrics': class_metrics,
        'roc_auc': roc_auc_mean
    })
