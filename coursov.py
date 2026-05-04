import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Настройка графиков
plt.style.use('ggplot')
sns.set_theme(style='whitegrid')
sns.set(font_scale=0.8)

print("\n[1] Загрузка данных...")
df = pd.read_csv('Disease and symptoms dataset.csv')

print(f"Размер данных: {df.shape}")
print(f"Количество заболеваний: {df['diseases'].nunique()}")
print(f"Количество симптомов: {df.shape[1] - 1}")

print("\n[2] Первичный анализ данных...")

# Проверка на пропуски
print(f"Пропуски в данных: {df.isnull().sum().sum()}")

# Анализ целевой переменной
disease_counts = df['diseases'].value_counts()
print(f"\nТоп-10 самых частых заболеваний:")
for disease, count in disease_counts.head(10).items():
    print(f"  {disease}: {count} записей ({count/len(df)*100:.2f}%)")

# Визуализация распределения заболеваний
plt.figure(figsize=(14, 6))
top_diseases = disease_counts.head(20)
sns.barplot(x=top_diseases.values, y=top_diseases.index, palette='viridis')
plt.title('Топ-20 заболеваний по частоте встречаемости')
plt.xlabel('Количество записей')
plt.tight_layout()
plt.savefig('disease_distribution.png', dpi=150)
plt.show()

# Анализ симптомов
symptom_cols = [col for col in df.columns if col != 'diseases']
symptom_freq = df[symptom_cols].sum().sort_values(ascending=False)

print(f"\nТоп-10 самых частых симптомов:")
for symptom, count in symptom_freq.head(10).items():
    print(f"  {symptom}: {count} раз ({count/len(df)*100:.2f}%)")

# Визуализация симптомов
plt.figure(figsize=(14, 6))
top_symptoms = symptom_freq.head(20)
sns.barplot(x=top_symptoms.values, y=top_symptoms.index, palette='rocket')
plt.title('Топ-20 самых частых симптомов')
plt.xlabel('Количество появлений')
plt.tight_layout()
plt.savefig('symptom_distribution.png', dpi=150)
plt.show()

# Анализ количества симптомов на заболевание
symptoms_per_disease = df[symptom_cols].sum(axis=1)
print(f"\nСтатистика по количеству симптомов на запись:")
print(f"  Минимум: {symptoms_per_disease.min()}")
print(f"  Максимум: {symptoms_per_disease.max()}")
print(f"  Среднее: {symptoms_per_disease.mean():.2f}")
print(f"  Медиана: {symptoms_per_disease.median()}")

# Гистограмма распределения
plt.figure(figsize=(10, 5))
plt.hist(symptoms_per_disease, bins=30, edgecolor='black', alpha=0.7)
plt.title('Распределение количества симптомов на одну запись')
plt.xlabel('Количество симптомов')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('symptoms_per_record.png', dpi=150)
plt.show()

print("\n[3] Подготовка данных...")

# Кодирование целевой переменной
le = LabelEncoder()
y = le.fit_transform(df['diseases'])
print(f"Количество классов: {len(le.classes_)}")

# Матрица признаков
X = df[symptom_cols].values
print(f"Размер матрицы признаков: {X.shape}")

# Разделение на обучающую (70%), валидационную (15%) и тестовую (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nРазмер выборок:")
print(f"  Обучающая: {X_train.shape[0]} записей ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Валидационная: {X_val.shape[0]} записей ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"  Тестовая: {X_test.shape[0]} записей ({X_test.shape[0]/len(df)*100:.1f}%)")

# Информация о сохранённых редких классах
print(f"\nВсе {len(le.classes_)} заболеваний сохранены в данных")
print("(включая встречающиеся только 1 раз)")

# Масштабирование (важно для SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("[4] МОДЕЛЬ 1: Случайный лес")

X_train_sampled = X_train
y_train_sampled = y_train

param_dist_rf = {
    'n_estimators': [50, 100],
    'max_depth': [20, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=2)  # НЕ используем все ядра!

random_rf = RandomizedSearchCV(rf_base, param_dist_rf, n_iter=4, cv=2,  # сильно уменьшено
                                scoring='accuracy', random_state=42, n_jobs=1, verbose=2)  # verbose=2 для детального вывода

print("Начало обучения (должно занять 1-2 минуты)...")
random_rf.fit(X_train_sampled, y_train_sampled)
print("Обучение завершено!")

print(f"Лучшие параметры: {random_rf.best_params_}")
print(f"Лучшая точность на кросс-валидации: {random_rf.best_score_:.4f}")

# Оценка на валидационной выборке
y_val_pred_rf = random_rf.predict(X_val)
val_accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
print(f"Точность на валидации: {val_accuracy_rf:.4f}")

# Оценка на тестовой выборке
y_test_pred_rf = random_rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
test_f1_rf = f1_score(y_test, y_test_pred_rf, average='weighted')
print(f"Точность на тесте: {test_accuracy_rf:.4f}")
print(f"F1-score (weighted) на тесте: {test_f1_rf:.4f}")

# Важность признаков
feature_importance = random_rf.best_estimator_.feature_importances_
top_features_idx = np.argsort(feature_importance)[-15:][::-1]

plt.figure(figsize=(12, 6))
plt.barh(range(15), feature_importance[top_features_idx][::-1])
plt.yticks(range(15), [symptom_cols[i] for i in top_features_idx[::-1]])
plt.xlabel('Важность признака')
plt.title('Топ-15 наиболее важных симптомов (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# Classification report для топ-10 классов
top_10_diseases = disease_counts.head(10).index
top_10_indices = [np.where(le.classes_ == d)[0][0] for d in top_10_diseases if d in le.classes_]

# Фильтруем тестовые данные для топ-10 классов
mask_top10 = np.isin(y_test, top_10_indices)

if mask_top10.sum() > 0:
    y_test_top10 = y_test[mask_top10]
    y_pred_top10_rf = y_test_pred_rf[mask_top10]

    # Получаем уникальные классы, которые реально присутствуют
    unique_labels = np.unique(y_test_top10)
    # Используем ТОЛЬКО те target_names, которые соответствуют реальным классам
    target_names_subset = [le.classes_[i] for i in unique_labels]

    print(f"\nКоличество классов в отфильтрованных данных: {len(unique_labels)}")
    print("Classification Report (топ-10 заболеваний по частоте) - Random Forest:")
    print(classification_report(y_test_top10, y_pred_top10_rf,
                                labels=unique_labels,
                                target_names=target_names_subset,
                                zero_division=0))

    # Матрица ошибок для топ-10 классов (только для тех, что реально есть)
    cm_rf = confusion_matrix(y_test_top10, y_pred_top10_rf, labels=unique_labels)
    cm_rf_norm = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(max(10, len(unique_labels) * 0.8), max(8, len(unique_labels) * 0.6)))
    sns.heatmap(cm_rf_norm, annot=True, fmt='.2f', cmap='Purples',
                xticklabels=target_names_subset,
                yticklabels=target_names_subset)
    plt.title('Матрица ошибок: Случайный лес (наиболее частые заболевания)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix_randomforest.png', dpi=150)
    plt.show()
else:
    print("В тестовой выборке нет топ-10 заболеваний для построения отчёта")

print("[5] МОДЕЛЬ 2: KNN (метод ближайших соседей)")

sample_size = 30000  # используем 30k образцов для обучения (из 170k)
indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
X_train_knn = X_train_scaled[indices]
y_train_knn = y_train[indices]
print(f"Для KNN используется выборка из {sample_size} записей (вместо {X_train_scaled.shape[0]})")

# Поиск лучшего k (количества соседей) с помощью кросс-валидации
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

knn_base = KNeighborsClassifier(n_jobs=2)
grid_knn = GridSearchCV(knn_base, param_grid_knn, cv=2, scoring='accuracy', n_jobs=1, verbose=1)

print("Начало обучения KNN (1-2 минуты)...")
grid_knn.fit(X_train_knn, y_train_knn)
print("Обучение KNN завершено!")

print(f"Лучшие параметры: {grid_knn.best_params_}")
print(f"Лучшая точность на кросс-валидации: {grid_knn.best_score_:.4f}")

y_val_pred_knn = grid_knn.predict(X_val_scaled)
val_accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
print(f"Точность на валидации: {val_accuracy_knn:.4f}")

y_test_pred_knn = grid_knn.predict(X_test_scaled)
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
test_f1_knn = f1_score(y_test, y_test_pred_knn, average='weighted')
print(f"Точность на тесте: {test_accuracy_knn:.4f}")
print(f"F1-score (weighted) на тесте: {test_f1_knn:.4f}")

# Classification report для топ-10 классов (KNN)
if mask_top10.sum() > 0:
    y_pred_top10_knn = y_test_pred_knn[mask_top10]

    print("\nClassification Report (топ-10 заболеваний по частоте) - KNN:")
    print(classification_report(y_test_top10, y_pred_top10_knn,
                                labels=unique_labels,
                                target_names=target_names_subset,
                                zero_division=0))

    # Матрица ошибок для топ-10 классов
    cm_knn = confusion_matrix(y_test_top10, y_pred_top10_knn, labels=unique_labels)
    cm_knn_norm = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(max(10, len(unique_labels) * 0.8), max(8, len(unique_labels) * 0.6)))
    sns.heatmap(cm_knn_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names_subset,
                yticklabels=target_names_subset)
    plt.title('Матрица ошибок: KNN (наиболее частые заболевания)')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix_knn.png', dpi=150)
    plt.show()

print("[6] СРАВНЕНИЕ РЕЗУЛЬТАТОВ")

results = pd.DataFrame({
    'Модель': ['Случайный лес', 'KNN'],
    'Точность на валидации': [val_accuracy_rf, val_accuracy_knn],
    'Точность на тесте': [test_accuracy_rf, test_accuracy_knn],
    'F1-score (тест)': [test_f1_rf, test_f1_knn]
})

print(results.to_string(index=False))

# Визуализация сравнения
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results['Модель']))
width = 0.25

bars1 = ax.bar(x - width, results['Точность на валидации'], width, label='Валидация', color='steelblue')
bars2 = ax.bar(x, results['Точность на тесте'], width, label='Тест', color='coral')
bars3 = ax.bar(x + width, results['F1-score (тест)'], width, label='F1-score', color='seagreen')

ax.set_xlabel('Модель')
ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение моделей: Случайный лес vs KNN')
ax.set_xticks(x)
ax.set_xticklabels(results['Модель'], rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=150)
plt.show()

print("[7] ВЫВОДЫ")

better_model = "Случайный лес" if test_accuracy_rf >= test_accuracy_knn else "KNN"
print(f"""
1. Случайный лес:
   - Преимущества: ансамблевый метод, устойчив к переобучению, оценивает важность признаков
   - Недостатки: медленнее на обучении, сложнее интерпретировать
   - Результат: точность {test_accuracy_rf:.4f}

2. KNN (метод ближайших соседей):
   - Преимущества: простой, не требует обучения (ленивый алгоритм), легко добавлять новые данные
   - Недостатки: медленный на больших данных (O(n²)), требует масштабирования
   - Результат: точность {test_accuracy_knn:.4f}
   - Лучший k = {grid_knn.best_params_['n_neighbors']}
   - Использована выборка из {sample_size} записей для ускорения

3. Лучшая модель: {better_model}

ПРИМЕЧАНИЕ: Все {len(le.classes_)} заболеваний сохранены в датасете.
""")

# Сохранение результатов
results.to_csv('model_comparison.csv', index=False)
print("\nРезультаты сохранены в 'model_comparison.csv'")