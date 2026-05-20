import pandas as pd
import os

# Реальные тестовые данные, на которых обучены модели
df = pd.read_csv('processed/splits/X_test_all.csv')
df_sample = df.sample(n=10000, random_state=42)

df_sample.to_csv('processed/test_data/isolation_forest_test_10k.csv', index=False)

print(f"Сохранено {len(df_sample)} строк")
if 'attack_class' in df_sample.columns:
    print(df_sample['attack_class'].value_counts())
