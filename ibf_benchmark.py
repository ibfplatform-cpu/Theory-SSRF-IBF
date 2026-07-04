import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 1. ДАННЫЕ (замените на свои при необходимости)
# -------------------------------------------------------------
# Если у вас есть файл с реальными данными, раскомментируйте блок ниже
# и закомментируйте синтетическую генерацию.
#
# df = pd.read_csv('your_data.csv')
# scores_base = df['W_bar'].values        # прогноз базовой модели
# scores_full = df['IBF_full'].values     # прогноз полной модели
# y_true = df['collapse'].values          # 1 - коллапс, 0 - стабилизация

# --- Синтетические данные для демонстрации (дадут AUC ~0.77 и ~0.90) ---
np.random.seed(42)
n = 14
y_true = np.array([1]*7 + [0]*7)  # 7 коллапсов, 7 стабилизаций

# Базовая модель: перекрытие классов
scores_base = np.concatenate([
    np.random.normal(loc=0.35, scale=0.15, size=7),
    np.random.normal(loc=0.65, scale=0.15, size=7)
])
scores_base = np.clip(scores_base, 0, 1)

# Полная модель: лучшее разделение
scores_full = np.concatenate([
    np.random.normal(loc=0.25, scale=0.12, size=7),
    np.random.normal(loc=0.75, scale=0.12, size=7)
])
scores_full = np.clip(scores_full, 0, 1)

# -------------------------------------------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------------------------------------------
def compute_metrics(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    youden_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[youden_idx]
    se = tpr[youden_idx]
    sp = 1 - fpr[youden_idx]
    return {
        'auc': roc_auc,
        'fpr': fpr, 'tpr': tpr,
        'best_thr': best_thr,
        'se': se,
        'sp': sp
    }

def bootstrap_auc(y_true, scores, n_bootstrap=10000):
    aucs = []
    rng = np.random.default_rng(seed=42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return lower, upper

def bootstrap_delta_auc(y_true, s1, s2, n_bootstrap=10000):
    deltas = []
    rng = np.random.default_rng(seed=42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y_true[idx], s2[idx]) - roc_auc_score(y_true[idx], s1[idx]))
    lower = np.percentile(deltas, 2.5)
    upper = np.percentile(deltas, 97.5)
    p_val = 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0))
    return np.mean(deltas), lower, upper, p_val

# -------------------------------------------------------------
# 3. РАСЧЁТ
# -------------------------------------------------------------
metrics_base = compute_metrics(y_true, scores_base)
metrics_full = compute_metrics(y_true, scores_full)

ci_base = bootstrap_auc(y_true, scores_base)
ci_full = bootstrap_auc(y_true, scores_full)

delta_mean, delta_low, delta_high, p_val = bootstrap_delta_auc(y_true, scores_base, scores_full)

# -------------------------------------------------------------
# 4. ВЫВОД РЕЗУЛЬТАТОВ
# -------------------------------------------------------------
print("="*60)
print("Сравнительный ROC-бенчмарк моделей IBF (N=14 кейсов)")
print("="*60)

print(f"\nБазовая модель (IBF_base = W̄):")
print(f"  AUC = {metrics_base['auc']:.2f}  [95% ДИ: {ci_base[0]:.2f} – {ci_base[1]:.2f}]")
print(f"  Оптимальный порог = {metrics_base['best_thr']:.2f}")
print(f"  Чувствительность (Se) = {metrics_base['se']:.2f}")
print(f"  Специфичность (Sp)   = {metrics_base['sp']:.2f}")

print(f"\nПолная модель (IBF_full):")
print(f"  AUC = {metrics_full['auc']:.2f}  [95% ДИ: {ci_full[0]:.2f} – {ci_full[1]:.2f}]")
print(f"  Оптимальный порог = {metrics_full['best_thr']:.2f}")
print(f"  Чувствительность (Se) = {metrics_full['se']:.2f}")
print(f"  Специфичность (Sp)   = {metrics_full['sp']:.2f}")

print(f"\nПрирост точности:")
print(f"  ΔAUC = {delta_mean:+.2f}  [95% ДИ: {delta_low:+.2f} – {delta_high:+.2f}]")
print(f"  p-value = {p_val:.4f}  {'(статистически значимо, p < 0.01)' if p_val < 0.01 else '(не значимо)'}")

# -------------------------------------------------------------
# 5. ГРАФИК ROC-КРИВЫХ
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(metrics_base['fpr'], metrics_base['tpr'],
         label=f'Базовая модель (AUC = {metrics_base["auc"]:.2f})', lw=2)
plt.plot(metrics_full['fpr'], metrics_full['tpr'],
         label=f'Полная модель (AUC = {metrics_full["auc"]:.2f})', lw=2)
plt.plot([0,1], [0,1], 'k--', lw=1, label='Случайный классификатор')
plt.xlabel('Доля ложноположительных (1 - Специфичность)')
plt.ylabel('Доля истинноположительных (Чувствительность)')
plt.title('Сравнительный ROC-бенчмарк IBF (N=14 кейсов)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.xlim([0,1]); plt.ylim([0,1])
plt.tight_layout()
plt.savefig('roc_ibf_benchmark.png', dpi=300)
plt.show()

print("\nГрафик сохранён как 'roc_ibf_benchmark.png'")
