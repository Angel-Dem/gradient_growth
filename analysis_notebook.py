"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ПРОГНОЗ РТО МАГАЗИНОВ ПЯТЁРОЧКА — ПОЛНЫЙ АНАЛИЗ                           ║
║  Хакатон Градиент роста (X5 Group × ВШЭ ФКН)                               ║
║                                                                              ║
║  Структура:                                                                  ║
║  0. Загрузка и первичный осмотр данных                                      ║
║  1. Анализ временных рядов: тренд, сезонность, распределение                ║
║  2. STL-разложение (Тренд + Сезонность + Шум)                               ║
║  3. Статистический анализ (АКФ, стационарность, корреляции)                 ║
║  4. Инжиниринг признаков (все типы из лекции)                               ║
║  5. Разбивка Train / Test                                                    ║
║  6. Линейная регрессия (Ridge) — базовый тест признаков                     ║
║  7. Gradient Boosting — финальная модель                                    ║
║  8. Сравнение моделей и важность признаков                                  ║
║  9. Финальное предсказание на ноябрь → submission_final.csv                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# 0. ИМПОРТЫ И ЗАГРУЗКА ДАННЫХ
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

print("=" * 60)
print("СТРУКТУРА ДАННЫХ")
print("=" * 60)
print(f"Строк:     {len(df):,}")
print(f"Месяцы:    {sorted(df['Месяц'].unique())}")
print(f"Магазинов: {df['new_id'].nunique():,}")
print()
print(df.dtypes)
print("\nПервые 3 строки:")
print(df.head(3).to_string())

# =============================================================================
# ОПИСАНИЕ ПРИЗНАКОВ
# =============================================================================
"""
ИНТЕРПРЕТАЦИЯ ДАННЫХ:
─────────────────────
• new_id       — уникальный ID магазина (20 615 магазинов)
• Месяц        — порядковый номер месяца (1=янв…10=окт), ЦЕЛЬ → 11=ноябрь
• РТО          — розничный товарооборот (выручка) ₽ — ЦЕЛЕВАЯ ПЕРЕМЕННАЯ

Статические характеристики магазина (не меняются по месяцам):
• Дата открытия, категориальный  — возраст: Новый / Средний / Давно открыт
• Торговая площадь, категориальный — Маленький / Средний / Большой / Очень большой
• Населенный пункт, Регион       — географическое расположение
• Численность населения          — жителей в городе/посёлке
• Количество домохозяйств        — семей в районе
• Трафик пеший/авто, в час       — поток людей рядом с магазином
• Маркетплейсы (100 м)           — конкуренция с онлайн-доставками
• Медицинские учреждения (300 м) — генератор трафика поблизости
• Школы (300 м)                  — генератор трафика (родители, дети)
• Остановки (300 м)              — доступность = больше покупателей
• Продуктовые магазины (500 м)   — конкурентная среда
• Пятерочки (500 м)              — конкуренция внутри сети (каннибализм)
• Количество касс                — прокси размера/пропускной способности
• Флаг алкогольной лицензии      — наличие = выше средний чек
"""

# =============================================================================
# 1. АНАЛИЗ ВРЕМЕННЫХ РЯДОВ: ТРЕНД, СЕЗОННОСТЬ, РАСПРЕДЕЛЕНИЕ
# =============================================================================
monthly = df.groupby('Месяц')['РТО'].agg(['mean','median',
    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
monthly.columns = ['Месяц','mean','median','q25','q75']

print("\n" + "=" * 60)
print("СРЕДНЕЕ РТО ПО МЕСЯЦАМ")
print("=" * 60)
for _, row in monthly.iterrows():
    print(f"  Месяц {int(row['Месяц']):2d}: {row['mean']/1e6:.1f}M  (медиана {row['median']/1e6:.1f}M)")

months = np.arange(1, 11)
MN = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт']

pivot = df.pivot(index='new_id', columns='Месяц', values='РТО')
cv_stores = (pivot.std(axis=1) / pivot.mean(axis=1)).values

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Рис. 1 — Анализ временных рядов РТО магазинов Пятёрочка', fontsize=14, fontweight='bold')

ax = axes[0,0]
ax.fill_between(monthly['Месяц'], monthly['q25']/1e6, monthly['q75']/1e6,
                alpha=0.25, color='steelblue', label='IQR (25–75%)')
ax.plot(monthly['Месяц'], monthly['mean']/1e6, 'o-', color='steelblue', lw=2.5, ms=7, label='Среднее РТО')
ax.plot(monthly['Месяц'], monthly['median']/1e6, 's--', color='coral', lw=2, ms=6, label='Медиана РТО')
ax.set_xticks(months); ax.set_xticklabels(MN)
ax.set_ylabel('РТО, млн руб.')
ax.set_title('Тренд РТО по месяцам (все магазины)\nIQR: крупные магазины сильно выделяются')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
for sid, col in zip([0, 100, 500, 1000, 5000, 10000], plt.cm.tab10(np.linspace(0, 1, 6))):
    s = df[df['new_id'] == sid].sort_values('Месяц')
    ax.plot(s['Месяц'], s['РТО']/1e6, 'o-', color=col, lw=1.8, ms=5, alpha=0.85, label=f'ID {sid}')
ax.set_xticks(months); ax.set_xticklabels(MN); ax.set_ylabel('РТО, млн руб.')
ax.set_title('Временные ряды отдельных магазинов\nКаждый имеет свой уровень → lag_1 очень важен')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1,0]
rto_vals = df['РТО'].values / 1e6
ax.hist(rto_vals, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(rto_vals), color='red', ls='--', lw=2, label=f'Среднее: {np.mean(rto_vals):.1f}M')
ax.axvline(np.median(rto_vals), color='orange', ls='--', lw=2, label=f'Медиана: {np.median(rto_vals):.1f}M')
ax.set_xlabel('РТО, млн руб.'); ax.set_ylabel('Кол-во записей')
ax.set_title('Распределение РТО\nПравосторонняя асимметрия — есть крупные магазины-выбросы')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.hist(cv_stores, bins=60, color='coral', alpha=0.75, edgecolor='white')
ax.axvline(np.median(cv_stores), color='red', ls='--', lw=2, label=f'Медиана CV = {np.median(cv_stores):.3f}')
ax.set_xlabel('Коэффициент вариации (std/mean)'); ax.set_ylabel('Кол-во магазинов')
ax.set_title(f'Волатильность РТО по магазинам\nCV~{np.median(cv_stores):.2f} → ряды стабильны, лаги работают хорошо')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_trends.png', dpi=120, bbox_inches='tight')
plt.show()

# =============================================================================
# 2. STL-РАЗЛОЖЕНИЕ: ТРЕНД + СЕЗОННОСТЬ + ШУМ
# =============================================================================
"""
STL-разложение: Y(t) = Тренд(t) + Сезонность(t) + Остаток(t)

Тренд     — долгосрочное направление (линейный рост/спад продаж)
Сезонность — циклические колебания (повторяются из года в год)
Остаток   — случайный шум (необъяснённая часть)

У нас только 10 месяцев (один неполный год), поэтому:
  - Тренд: линейная регрессия РТО ~ Месяц (МНК)
  - Сезонность: отклонение факта от тренда, сглаженное скользящим средним
  - Остаток: Y - Тренд - Сезонность

КАК ИСПОЛЬЗОВАТЬ В МОДЕЛИ:
  Если вычесть тренд → предсказываем только колебания (десезонализация).
  В нашем случае lag_1 несёт всю информацию об уровне магазина автоматически,
  поэтому ручное вычитание необязательно.
  НО: признак trend_slope (наклон ряда) добавляем явно — это важно!
"""

monthly_mean = df.groupby('Месяц')['РТО'].mean().values
trend_coef = np.polyfit(months, monthly_mean, 1)
trend = np.polyval(trend_coef, months)

# Сезонность = факт - тренд, сглаженная скользящим средним
detrended = monthly_mean - trend
def moving_avg(x, w=3):
    r = np.convolve(x, np.ones(w)/w, mode='same')
    r[0] = x[0]; r[-1] = x[-1]
    return r
seasonal = moving_avg(detrended, w=3)
residual = monthly_mean - trend - seasonal

fig, axes = plt.subplots(4, 1, figsize=(14, 14))
fig.suptitle('Рис. 2 — STL-разложение: средний РТО по всем магазинам', fontsize=13, fontweight='bold')

axes[0].plot(months, monthly_mean/1e6, 'o-', color='steelblue', lw=2.5, ms=8)
axes[0].fill_between(months, monthly_mean/1e6, alpha=0.12, color='steelblue')
axes[0].set_title('① Исходный ряд Y(t) — что мы наблюдаем', fontweight='bold')
axes[0].set_ylabel('РТО, млн ₽'); axes[0].set_xticks(months)
axes[0].set_xticklabels(MN); axes[0].grid(alpha=0.3)

axes[1].plot(months, trend/1e6, 'o-', color='darkorange', lw=2.5, ms=8, label='Тренд (МНК)')
axes[1].plot(months, monthly_mean/1e6, '--', color='steelblue', lw=1.5, alpha=0.4, label='Факт')
axes[1].set_title(f'② Тренд T(t) — рост ≈ +{trend_coef[0]/1e6:.2f} млн руб./мес '
                  f'({trend[0]/1e6:.1f}M → {trend[-1]/1e6:.1f}M)', fontweight='bold')
axes[1].set_ylabel('РТО, млн ₽'); axes[1].set_xticks(months)
axes[1].set_xticklabels(MN); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].bar(months, seasonal/1e6, color=['#2ca02c' if v>=0 else '#d62728' for v in seasonal],
            alpha=0.8, edgecolor='white')
axes[2].axhline(0, color='black', lw=1.2)
for m, v in zip(months, seasonal/1e6):
    axes[2].annotate(f'{v:+.1f}M', (m, v), ha='center', va='bottom' if v>=0 else 'top', fontsize=9)
axes[2].set_title('③ Сезонность S(t) — циклические отклонения от тренда\n'
                  '   Март↑ пик покупок, Сентябрь↓ спад (конец лета)', fontweight='bold')
axes[2].set_ylabel('Откл., млн ₽'); axes[2].set_xticks(months)
axes[2].set_xticklabels(MN); axes[2].grid(alpha=0.3, axis='y')

axes[3].bar(months, residual/1e6, color='mediumpurple', alpha=0.75, edgecolor='white')
axes[3].axhline(0, color='black', lw=1.2)
axes[3].axhline(residual.std()/1e6, color='red', ls='--', lw=1.5, label=f'+1σ = {residual.std()/1e6:.2f}M')
axes[3].axhline(-residual.std()/1e6, color='red', ls='--', lw=1.5, label=f'-1σ = {-residual.std()/1e6:.2f}M')
axes[3].set_title('④ Остаток R(t) = Y - T - S (шум)\n'
                  '   Маленький шум → данные хорошо объясняются трендом + сезонностью', fontweight='bold')
axes[3].set_ylabel('Остаток, млн ₽'); axes[3].set_xticks(months)
axes[3].set_xticklabels(MN); axes[3].legend(); axes[3].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig2_decomposition.png', dpi=120, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ: АКФ, СТАЦИОНАРНОСТЬ, ДИСПЕРСИЯ, КОРРЕЛЯЦИИ
# =============================================================================
"""
АКФ — Автокорреляционная функция:
  Показывает, насколько значение ряда в момент t коррелирует с t-lag.
  Высокая АКФ(1) → lag_1 будет отличным признаком модели.
  Если АКФ убывает медленно → ряд нестационарен (есть тренд).

Стационарность:
  Стационарный ряд: среднее и дисперсия не зависят от времени.
  Нестационарный: есть тренд → нужно учитывать в модели.
  Прокси: корреляция Спирмена РТО ~ Месяц. Если >> 0 → тренд вверх.

Корреляция признаков с РТО (Пирсон):
  Линейная связь каждого признака с целевой переменной.
  → Кол-во касс (r=0.64) — ключевой экзогенный предиктор!
"""

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Рис. 3 — Статистический анализ временных рядов РТО', fontsize=14, fontweight='bold')

# АКФ
ax = axes[0,0]
conf = 1.96 / np.sqrt(10)
acf = [1.0]
for lag in range(1, 9):
    acf.append(np.corrcoef(monthly_mean[:-lag], monthly_mean[lag:])[0,1])
ax.bar(range(9), acf, color=['steelblue' if v>=0 else 'coral' for v in acf], alpha=0.8, edgecolor='white')
ax.axhline(conf, color='red', ls='--', lw=1.5, label=f'95% дов. инт. ±{conf:.2f}')
ax.axhline(-conf, color='red', ls='--', lw=1.5); ax.axhline(0, color='black', lw=1)
ax.set_xlabel('Лаг (мес.)'); ax.set_ylabel('Автокорреляция')
ax.set_title('АКФ среднего РТО\nВысокая АКФ(1) → lag_1 — ключевой признак!')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Скорость изменения (первые разности)
ax = axes[0,1]
diffs = np.diff(monthly_mean)
ax.bar(np.arange(2, 11), diffs/1e6,
       color=['#2ca02c' if d>=0 else '#d62728' for d in diffs], alpha=0.8, edgecolor='white')
ax.axhline(0, color='black', lw=1.2)
for m, d in zip(np.arange(2, 11), diffs/1e6):
    ax.annotate(f'{d:+.1f}M', (m, d), ha='center', va='bottom' if d>=0 else 'top', fontsize=9)
ax.set_xticks(np.arange(2, 11)); ax.set_xticklabels(MN[1:])
ax.set_title('Скорость изменения: первые разности Δ(t)\n'
             'diff1 = RTO(t-1)-RTO(t-2) улавливает ускорение/торможение')
ax.set_ylabel('ΔРТО, млн ₽'); ax.grid(alpha=0.3, axis='y')

# Тест на стационарность
ax = axes[0,2]
rto_arr = pivot.values
sp_corrs = np.array([stats.spearmanr(range(10), rto_arr[i])[0] for i in range(len(rto_arr))])
ax.hist(sp_corrs, bins=50, color='steelblue', alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', lw=1.5)
ax.axvline(np.mean(sp_corrs), color='red', ls='--', lw=2, label=f'Среднее: {np.mean(sp_corrs):.3f}')
pup = (sp_corrs > 0.5).mean()*100; pdn = (sp_corrs < -0.5).mean()*100
ax.set_title(f'Стационарность (корреляция Спирмена РТО~Месяц)\n'
             f'↑ тренд {pup:.0f}%,  ↓ тренд {pdn:.0f}%,  без тренда {100-pup-pdn:.0f}%')
ax.set_xlabel('Корреляция Спирмена'); ax.legend(); ax.grid(alpha=0.3)

# Дисперсия по месяцам
ax = axes[1,0]
mstd = df.groupby('Месяц')['РТО'].std().values
mm   = df.groupby('Месяц')['РТО'].mean().values
ax.bar(months, mstd/1e6, color='mediumpurple', alpha=0.75, edgecolor='white', label='std РТО')
ax2 = ax.twinx()
ax2.plot(months, (mstd/mm)*100, 'o--', color='red', lw=2, label='CV, %')
ax.set_xticks(months); ax.set_xticklabels(MN)
ax.set_ylabel('Стд. откл., млн ₽', color='mediumpurple')
ax2.set_ylabel('CV, %', color='red')
ax.set_title('Дисперсия РТО по месяцам\nCV почти одинаков → умеренная гетероскедастичность')
l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=9); ax.grid(alpha=0.3)

# Корреляция экзогенных признаков с РТО
ax = axes[1,1]
num_cols = ['Численность населения','Количество домохозяйств','Трафик пеший, в час',
            'Трафик авто, в час','Маркетплейсы, доставки, постаматы (100 м)',
            'Медицинские уч. и аптеки (300 м)','Школы (300 м)','Остановки (300 м)',
            'Продуктовые магазины (500 м)','Пятерочки (500 м)','Количество касс','Флаг алкогольной лицензии']
corrs = df[num_cols + ['РТО']].corr()['РТО'].drop('РТО')
snames = ['Население','Домохозяйства','Пеш.трафик','Авт.трафик','Маркетпл.',
          'Медучр.','Школы','Остановки','Прод.маг.','Пятёрочки','Кол.касс','Алкоголь']
brs = ax.barh(snames, corrs.values,
              color=['steelblue' if c>=0 else 'coral' for c in corrs.values], alpha=0.8, edgecolor='white')
ax.axvline(0, color='black', lw=1.2)
for b, v in zip(brs, corrs.values):
    ax.text(v+(0.01 if v>=0 else -0.01), b.get_y()+b.get_height()/2,
            f'{v:.3f}', va='center', ha='left' if v>=0 else 'right', fontsize=8)
ax.set_xlabel('Корреляция Пирсона с РТО')
ax.set_title('Корреляция экзогенных признаков с РТО\n'
             'Кол-во касс (0.64) — лучший статический предиктор!'); ax.grid(alpha=0.3, axis='x')

# Волатильность по магазинам
ax = axes[1,2]
ax.hist(cv_stores, bins=60, color='coral', alpha=0.75, edgecolor='white')
ax.axvline(np.median(cv_stores), color='red', ls='--', lw=2, label=f'Медиана CV={np.median(cv_stores):.3f}')
ax.axvline(np.mean(cv_stores), color='darkred', ls='-', lw=1.5, label=f'Среднее CV={np.mean(cv_stores):.3f}')
ax.set_xlabel('Коэффициент вариации (std/mean)'); ax.set_ylabel('Кол-во магазинов')
ax.set_title(f'Волатильность по магазинам\nCV~{np.median(cv_stores):.2f}: ряды стабильны → лаги работают')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_stats.png', dpi=120, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. ИНЖИНИРИНГ ПРИЗНАКОВ
# =============================================================================
"""
Все признаки строятся на данных ДО целевого месяца (нет утечки).

1. ЛАГОВЫЕ ПРИЗНАКИ
   lag1 = РТО(t-1)  → самый важный: каждый магазин почти не меняет уровень
   lag2 = РТО(t-2)  → краткосрочные колебания
   lag3 = РТО(t-3)  → сезонные паттерны
   (lag12 = год назад недоступен — у нас только 10 месяцев)

2. СКОЛЬЗЯЩИЕ СРЕДНИЕ (сглаживают шум, дают стабильный уровень)
   ma3   = среднее за последние 3 месяца
   ma6   = среднее за последние 6 месяцев
   ma_all = среднее за все доступные месяцы

3. СТАНДАРТНОЕ ОТКЛОНЕНИЕ И CV (волатильность магазина)
   std_all = насколько нестабильна выручка
   cv      = std/mean — нормализованная нестабильность

4. СКОРОСТЬ ИЗМЕНЕНИЯ РЯДА (первые разности = аналог производной)
   diff1  = RTO(t-1) - RTO(t-2) — прирост/убыль в последний месяц
   growth = diff1 / RTO(t-2) — темп роста в %

5. СЕЗОННЫЙ ПРИЗНАК
   ratio_last = RTO(t-1) / ma_all
   Если > 1 → магазин "выше нормы" → ожидаем рост и в следующем месяце

6. ТРЕНД-ПРИЗНАК (наклон ряда магазина)
   trend      = наклон линейного тренда (через ковариацию, быстро)
   trend_norm = trend / ma_all — нормированный темп роста

7. КВАНТИЛЬНЫЕ СТАТИСТИКИ (устойчивы к выбросам)
   med = медиана РТО магазина
   rng = размах (max - min)

8. ЭКЗОГЕННЫЕ ПРИЗНАКИ МАГАЗИНА (статика)
   Кол-во касс (r=0.64), регион, площадь, трафик и т.д.
"""

STATIC_FEATS = [
    'open_enc', 'size_enc', 'city_enc', 'region_enc',
    'Численность населения', 'Количество домохозяйств',
    'Трафик пеший, в час', 'Трафик авто, в час',
    'Маркетплейсы, доставки, постаматы (100 м)',
    'Медицинские уч. и аптеки (300 м)', 'Школы (300 м)',
    'Остановки (300 м)', 'Продуктовые магазины (500 м)',
    'Пятерочки (500 м)', 'Количество касс', 'Флаг алкогольной лицензии'
]


def build_features(data, target_month):
    """
    Строим матрицу признаков для предсказания РТО в месяц target_month.
    Используем ТОЛЬКО данные месяцев 1 ... (target_month - 1) — нет утечки!
    """
    avail = [f'rto_m{m}' for m in range(1, target_month)]
    rto = data[avail].values  # shape: (n_stores, n_months_available)

    f = {}

    # 1. Лаговые признаки
    f['lag1'] = data[f'rto_m{target_month-1}'].values
    f['lag2'] = data[f'rto_m{target_month-2}'].values if target_month >= 3 else np.zeros(len(data))
    f['lag3'] = data[f'rto_m{target_month-3}'].values if target_month >= 4 else np.zeros(len(data))

    # 2. Скользящие средние
    f['ma3']    = rto[:, -3:].mean(1) if rto.shape[1] >= 3 else rto.mean(1)
    f['ma6']    = rto[:, -6:].mean(1) if rto.shape[1] >= 6 else rto.mean(1)
    f['ma_all'] = rto.mean(1)

    # 3. Стд. отклонение и CV
    f['std_all'] = rto.std(1)
    f['cv']      = f['std_all'] / (f['ma_all'] + 1e-9)

    # 4. Скорость изменения (разности)
    if target_month >= 3:
        f['diff1']  = (data[f'rto_m{target_month-1}'] - data[f'rto_m{target_month-2}']).values
        f['growth'] = f['diff1'] / (np.abs(f['lag2']) + 1e-9)
    else:
        f['diff1']  = np.zeros(len(data))
        f['growth'] = np.zeros(len(data))

    # 5. Сезонный признак
    f['ratio_last'] = f['lag1'] / (f['ma_all'] + 1e-9)

    # 6. Наклон тренда (ковариационный метод — быстро, без цикла)
    x = np.arange(rto.shape[1], dtype=float)
    x -= x.mean()  # центрируем x
    f['trend']      = (rto * x).sum(1) / ((x**2).sum() + 1e-9)
    f['trend_norm'] = f['trend'] / (f['ma_all'] + 1e-9)

    # 7. Квантильные статистики
    f['med'] = np.median(rto, 1)
    f['rng'] = rto.max(1) - rto.min(1)

    # 8. Экзогенные признаки магазина
    for col in STATIC_FEATS:
        f[col] = data[col].values

    return pd.DataFrame(f, index=data.index)


# =============================================================================
# 5. РАЗБИВКА TRAIN / TEST (Walk-Forward Validation)
# =============================================================================
"""
Walk-Forward Validation — стандарт для временных рядов:

    Train:   предсказываем месяцы 4–9 (история 1..target-1 для каждого)
    Test:    предсказываем месяц 10 (октябрь) — честная оценка качества
    Submit:  предсказываем месяц 11 (ноябрь) — финальный прогноз

Почему нельзя случайно перемешать 80/20?
    Временные ряды нельзя перемешивать! Если дать модели "будущее" при обучении
    → утечка данных → завышенные метрики → плохой результат на тесте.
"""

static_df = df[df['Месяц'] == 1][
    ['new_id', 'Дата открытия, категориальный', 'Торговая площадь, категориальный',
     'Населенный пункт', 'Регион', 'Численность населения', 'Количество домохозяйств',
     'Трафик пеший, в час', 'Трафик авто, в час',
     'Маркетплейсы, доставки, постаматы (100 м)',
     'Медицинские уч. и аптеки (300 м)', 'Школы (300 м)', 'Остановки (300 м)',
     'Продуктовые магазины (500 м)', 'Пятерочки (500 м)',
     'Количество касс', 'Флаг алкогольной лицензии']
].set_index('new_id')

pivot2 = df.pivot(index='new_id', columns='Месяц', values='РТО')
pivot2.columns = [f'rto_m{c}' for c in pivot2.columns]
data = static_df.join(pivot2)

for col, nm in [('Дата открытия, категориальный', 'open'),
                ('Торговая площадь, категориальный', 'size'),
                ('Населенный пункт', 'city'),
                ('Регион', 'region')]:
    data[f'{nm}_enc'] = LabelEncoder().fit_transform(data[col])

COLS = list(build_features(data, 10).columns)  # канонический порядок признаков

# Обучающая выборка: walk-forward по месяцам 4..9
print("Строим Train (предсказания месяцев 4–9)...")
X_parts, y_parts = [], []
for target_m in range(4, 10):
    X_parts.append(build_features(data, target_m)[COLS])
    y_parts.append(data[f'rto_m{target_m}'].values)

X_train = pd.concat(X_parts).reset_index(drop=True)
y_train = np.concatenate(y_parts)

# Тестовая выборка: месяц 10 (предсказываем по истории 1..9)
X_test = build_features(data, 10)[COLS]
y_test = data['rto_m10'].values

print(f"Train: {X_train.shape[0]:,} строк × {X_train.shape[1]} признаков")
print(f"Test:  {X_test.shape[0]:,} строк  (20 615 магазинов, месяц 10)")

# =============================================================================
# 6. ЛИНЕЙНАЯ РЕГРЕССИЯ (RIDGE)
# =============================================================================
"""
Ridge = линейная регрессия с L2-регуляризацией.
Штрафует большие коэффициенты → борьба с мультиколлинеарностью (lag1, ma3, ma6 коррелируют).

Зачем тестировать перед GBM?
  1. Быстро проверить: несут ли признаки линейную информацию?
  2. Если Ridge тоже хороший → данные линейно сепарабельны
  3. Лекция: "хорошие признаки делают хорошей даже линейную модель"
  4. Ridge можно использовать как базовый уровень для ансамбля

ВАЖНО: Ridge требует нормализации! Признаки на разных масштабах
(lag1 ~ 10^7, флаг алкоголя ~ 0/1) → нужен StandardScaler.
"""
scaler = StandardScaler()
ridge = Ridge(alpha=100)
ridge.fit(scaler.fit_transform(X_train), y_train)
pred_ridge = ridge.predict(scaler.transform(X_test))
mape_ridge = 100 * np.mean(np.abs((pred_ridge - y_test) / y_test))

print(f"\nRidge MAPE на тесте (m10): {mape_ridge:.2f}%  → {100-mape_ridge:.2f} баллов")
ridge_coefs = pd.Series(np.abs(ridge.coef_), index=COLS).sort_values(ascending=False)
print("Топ-10 признаков (|коэффициент| Ridge):")
print(ridge_coefs.head(10).to_string())

# =============================================================================
# 7. GRADIENT BOOSTING
# =============================================================================
"""
GradientBoostingRegressor — ансамбль деревьев решений, обучаемых последовательно.
Каждое дерево исправляет ошибки предыдущих (градиентный спуск в пространстве функций).

Преимущества над Ridge:
  1. Улавливает нелинейные паттерны (взаимодействия lag1 × регион × касс)
  2. Устойчив к выбросам (крупные магазины не сильно влияют)
  3. Автоматически выбирает важные признаки
  4. Не требует нормализации

Гиперпараметры:
  n_estimators=300  — кол-во деревьев (больше = точнее, но дольше)
  max_depth=5       — глубина дерева (5 = хороший баланс)
  learning_rate=0.05 — мал. шаг + много деревьев = устойчивее
  min_samples_leaf=10 — L2-регуляризация через мин. объектов в листе
  subsample=0.8     — 80% объектов на каждое дерево (Stochastic GB)
"""
print("\nОбучаем Gradient Boosting...")
gbm = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gbm.fit(X_train, y_train)
pred_gbm = gbm.predict(X_test)
mape_gbm = 100 * np.mean(np.abs((pred_gbm - y_test) / y_test))
print(f"GBM MAPE на тесте (m10): {mape_gbm:.2f}%  → {100-mape_gbm:.2f} баллов")

pred_base = data['rto_m9'].values
mape_base = 100 * np.mean(np.abs((pred_base - y_test) / y_test))
print(f"Бейзлайн (m10=m9) MAPE:  {mape_base:.2f}%  → {100-mape_base:.2f} баллов")

# =============================================================================
# 8. ВАЖНОСТЬ ПРИЗНАКОВ И СРАВНЕНИЕ МОДЕЛЕЙ
# =============================================================================
feat_imp = pd.Series(gbm.feature_importances_, index=COLS).sort_values(ascending=False)
print("\nТоп-10 важных признаков (GBM):")
print(feat_imp.head(10).to_string())

feat_name_map = {
    'lag1':'Лаг 1 (прошлый мес.)', 'lag2':'Лаг 2', 'lag3':'Лаг 3',
    'ma3':'Скол. среднее 3M', 'ma6':'Скол. среднее 6M', 'ma_all':'Среднее по всем',
    'std_all':'Ст. откл.', 'cv':'Коэф. вариации CV',
    'diff1':'Скорость Δ1 (разность)', 'growth':'Темп роста %',
    'ratio_last':'Посл./Среднее (сезон)', 'trend':'Наклон тренда',
    'trend_norm':'Норм. наклон тренда', 'med':'Медиана', 'rng':'Размах',
    'open_enc':'Тип (возраст открытия)', 'size_enc':'Торг. площадь',
    'city_enc':'Населённый пункт', 'region_enc':'Регион',
    'Количество касс':'Кол-во касс', 'Численность населения':'Числ. населения',
}

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.suptitle('Рис. 4 — Важность признаков и сравнение моделей', fontsize=14, fontweight='bold')

ax = axes[0]
top = feat_imp.head(20)
names = [feat_name_map.get(f, f) for f in top.index]
clrs = plt.cm.RdYlGn(np.linspace(0.25, 0.92, len(top)))[::-1]
ax.barh(names[::-1], top.values[::-1]*100, color=clrs, edgecolor='white', alpha=0.88)
ax.set_xlabel('Важность признака, %')
ax.set_title('Топ-20 признаков по важности\n(GradientBoosting Feature Importance)')
ax.grid(alpha=0.3, axis='x')
for i, v in enumerate(top.values[::-1]*100):
    ax.text(v+0.1, i, f'{v:.1f}%', va='center', fontsize=8)

ax = axes[1]
mapes = [mape_base, mape_ridge, mape_gbm]
scores = [max(0, 100-m) for m in mapes]
bars = ax.bar(['Простой\nбейзлайн\n(m10=m9)', 'Ridge\n(линейная)', 'Gradient\nBoosting'],
              scores, color=['#d62728','#ff7f0e','#2ca02c'], alpha=0.85, edgecolor='white', width=0.55)
ax.axhline(54.34, color='gray', ls='--', lw=2, label='Планка конкурса (54.34)')
for bar, sc, mp in zip(bars, scores, mapes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{sc:.1f} балл.\nMAPE={mp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Баллы (100 − MAPE)'); ax.set_ylim(0, 112)
ax.set_title('Сравнение моделей\n(валидация: месяц 10 = октябрь)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig4_models.png', dpi=120, bbox_inches='tight')
plt.show()

# Scatter: факт vs прогноз
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Рис. 5 — Качество предсказаний: факт vs прогноз (месяц 10)', fontsize=13, fontweight='bold')
for ax, pred, name, mape, col in zip(axes,
        [pred_ridge, pred_gbm], ['Ridge','GradientBoosting'], [mape_ridge, mape_gbm],
        ['steelblue', '#2ca02c']):
    lim = max(y_test.max(), pred.max()) / 1e6
    ax.scatter(y_test/1e6, pred/1e6, alpha=0.08, s=3, color=col)
    ax.plot([0, lim], [0, lim], 'r--', lw=2, label='Идеальная y=x')
    ax.set_xlabel('Факт РТО, млн руб.'); ax.set_ylabel('Прогноз РТО, млн руб.')
    ax.set_title(f'{name}: MAPE = {mape:.1f}%  →  {100-mape:.1f} баллов')
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_scatter.png', dpi=120, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. ФИНАЛЬНОЕ ПРЕДСКАЗАНИЕ НА НОЯБРЬ (МЕСЯЦ 11) → submission_final.csv
# =============================================================================
"""
Для ноября используем всю историю янв–окт (m1..m10):
  lag1      = РТО за октябрь
  ma3       = среднее авг–окт
  trend     = наклон по всем 10 точкам
  и т.д.

Финальный прогноз = ансамбль Ridge + GBM:
  pred_final = 0.3 * Ridge + 0.7 * GBM
  GBM сильнее по точности, Ridge добавляет устойчивость к экстремальным магазинам.
"""
X_nov = build_features(data, 11)[COLS]
pred_nov_ridge = ridge.predict(scaler.transform(X_nov))
pred_nov_gbm   = gbm.predict(X_nov)
pred_nov_final = 0.3 * pred_nov_ridge + 0.7 * pred_nov_gbm  # ансамбль

submission = pd.DataFrame({'new_id': data.index, 'rto': pred_nov_final})
submission.to_csv('submission_final.csv', index=False)

print("\n" + "=" * 60)
print("ФИНАЛЬНОЕ ПРЕДСКАЗАНИЕ НОЯБРЬ")
print("=" * 60)
print(f"Строк: {len(submission)}")
print(f"Среднее РТО: {pred_nov_final.mean()/1e6:.2f} млн руб.")
print(f"Медиана РТО: {np.median(pred_nov_final)/1e6:.2f} млн руб.")
print(f"Сохранено: submission_final.csv")
print(submission.head(10).to_string())
