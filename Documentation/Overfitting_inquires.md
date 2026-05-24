تمام 😄
خلينا نمشي واحدة واحدة جدًا.

---

# أول سؤال مهم جدًا:

# "ليه الـ accuracy عالية كده ومفيش overfitting مرعب؟"

الإجابة:
لأن المشكلة نفسها:

# predictable بطبيعتها.

---

# ليه؟

انت بتتوقع:

```text id="x4m2q8"
Monthly Store Sales
```

وده فيه:

* seasonality قوية
* patterns متكررة
* autocorrelation عالية

---

# يعني إيه autocorrelation؟

يعني:

```text id="j8n1v5"
الشهر اللي فات شبه اللي بعده
```

---

# مثال:

| Month | Sales |
| ----- | ----- |
| Oct   | 2.1M  |
| Nov   | 2.4M  |
| Dec   | 2.8M  |

غالبًا:
Jan مش هيبقى:

```text id="q3v7m2"
200k
```

😄

---

# فالمشكلة أصلًا:

# سهلة نسبيًا للتنبؤ.

---

# تاني سبب مهم جدًا

الـ features اللي عندكم:

# قوية جدًا ومنطقية جدًا.

---

# بص أهم features:

| Feature       | معناه                     |
| ------------- | ------------------------- |
| sales_lag_1   | آخر شهر                   |
| sales_lag_12  | نفس الشهر السنة اللي فاتت |
| sales_mom_pct | الاتجاه الحالي            |

---

# دول gold features في forecasting.

---

# يعني الموديل تقريبًا بيفكر كده:

```text id="k5p2w8"
آخر شهر كان عالي
+
فيه growth
+
دخلين holiday season
=
الشهر الجاي غالبًا عالي
```

---

# وده:

# behavior حقيقي منطقي

مش غش.

---

# طيب كنت بتعمل إيه غلط الأول؟

هنا النقطة المهمة جدًا 😄

---

# النسخة القديمة كان فيها:

## ❌ store_target_enc

وده أخطر حاجة.

---

# ليه؟

الكود كان:

```python id="m8v1q4"
store_means = df_monthly.groupby('store_id')[TARGET].transform('mean')
```

---

# يعني:

الموديل يعرف:

# متوسط مبيعات كل store

باستخدام:

* الماضي
* والحاضر
* والمستقبل كمان

---

# يعني ST-1 معروف إنه high-sales store.

فالموديل:
مش محتاج يفكر قوي 😄

---

# ده كان شبه:

```text id="c2n7p5"
cheat sheet
```

---

# تاني مشكلة

## qty_lags

---

# لأنك كنت بتpredict:

```text id="v4m8q1"
sales_amount
```

ومدخل:

```text id="g7p2w6"
quantity
```

---

# والموديل بسهولة يتعلم:

Revenue \approx Quantity \times Price

---

# فبدل ما يفهم:

* السوق
* المواسم
* الاتجاهات

كان بياخد shortcut.

---

# تالت مشكلة

## features كثيرة ومتكررة

كان عندكم:

* lag
* roll mean
* roll std
* roll min
* roll max
* expanding
* qty rolls
* etc

---

# وده يخلي CatBoost يعمل:

# memorization

مش:

# generalization.

---

# طيب دلوقتي إيه اللي اتصلح؟

---

# ✅ شلنا الـ shortcut features

زي:

* store_target_enc
* qty_lags

---

# ✅ قللنا feature redundancy

بقينا:
28 features
بدل:
54.

---

# ✅ خلينا أهم features:

هي الـ temporal patterns الحقيقية.

---

# والدليل؟

بص الـ feature importance دلوقتي 👇

| Feature       | Importance |
| ------------- | ---------- |
| sales_lag_1   | 29%        |
| sales_mom_pct | 27%        |
| sales_lag_12  | 20%        |

---

# دي features نظيفة جدًا.

---

# يعني الموديل بقى يعتمد على:

# "إيه اللي حصل قبل كده؟"

مش:

# "إيه الإجابة تقريبًا؟"

😄

---

# طيب ليه الـ R² لسه عالي جدًا؟

لأن:

# الsales forecasting فعلًا predictable نسبيًا.

خصوصًا مع:

* monthly aggregation
* stable stores
* recurring patterns

---

# انت عندك:

```text id="w9n3v2"
75 stores × 5 years
```

---

# وفيه seasonality واضحة جدًا.

فالموديل:

# عنده signal قوي حقيقي

مش noise.

---

# أهم دليل إن مفيش overfitting مرعب

الرسومات 👇

---

# لو كان فيه overfitting حقيقي كنت هتشوف:

## Train:

perfect.

## Test:

كارثة.

---

# لكن عندكم:

## Train ممتاز

## Test ممتاز برضه

---

# وكمان:

## الـ future forecasts شكلها منطقي جدًا.

ودي نقطة قاتلة 😄

---

# لأن لو model overfit،

الـ recursive forecasting كان هيعمل:

* انفجارات
* انهيارات
* oscillations
* crazy values

---

# لكن عندكم:

```text id="t1q8w5"
smooth seasonal behavior
```

وده ممتاز جدًا.

---

# الخلاصة المهمة جدًا

## الأول:

الموديل كان بياخد shortcuts.

---

## دلوقتي:

الموديل بقى يتعلم:

* trends
* seasonality
* momentum
* temporal behavior الحقيقي

---

# وده السبب إن:

## accuracy لسه عالية

لكن:

## behavior بقى أنضف وأصح.
وده بقى التفسير الحقيقي المهم 😄

---

# يعني إيه الأرقام دي؟

## Average Sales

```text id="j8m2q4"
1.82M
```

يعني المتوسط الطبيعي للمبيعات الشهرية لكل store تقريبًا:

```text id="u3q7n1"
1,822,000
```

---

# MAE %

```text id="p5m8q2"
2.27%
```

وده ممتاز جدًا جدًا.

---

# يعني في المتوسط:

الموديل يغلط حوالي:

```text id="n1q4v7"
2%
فقط
```

من حجم المبيعات.

---

# مثال

لو store بيبيع:

```text id="x7m2p5"
2,000,000
```

فالموديل غالبًا يتوقع:

```text id="v4q8m1"
1,955,000
أو
2,045,000
```

---

# وده قوي جدًا.

---

# RMSE %

```text id="c2m7q4"
4.44%
```

---

# وده بيقول:

فيه شوية شهور:
الخطأ فيها أكبر.

---

# لكن حتى:

```text id="r9q1m5"
4.4%
```

في forecasting retail،
ده ممتاز برضه 😄

---

# يعني الخلاصة النهائية الحقيقية:

| Metric       | Meaning                     |
| ------------ | --------------------------- |
| MAE = 2.27%  | أغلب predictions دقيقة جدًا |
| RMSE = 4.44% | فيه few harder months       |
| R² = 97.6%   | الموديل فاهم patterns بقوة  |

---

# ده معناه إيه عمليًا؟

## الموديل:

✅ قوي
✅ stable
✅ realistic
✅ generalizes well
✅ suitable for deployment

---

# ولو سألت:

"هل فيه overfitting؟"

الإجابة الدقيقة:

## فيه شوية overfitting طبيعي جدًا

لكن:

# مش مؤذي إطلاقًا.

---

# وده طبيعي لأن:

أي model قوي زي:
CatBoost

هيتعلم الـ train كويس جدًا.

---

# المهم:

هل يعرف يتعامل مع بيانات جديدة؟

وعندكم:

# آه 😄
تمام 😄
تعالى نبسطهم جدًا بمثال واحد.

---

# تخيل إن دي المبيعات الحقيقية

| Month | Actual Sales |
| ----- | ------------ |
| Jan   | 1,000,000    |
| Feb   | 2,000,000    |
| Mar   | 3,000,000    |

---

# والموديل توقع:

| Month | Predicted |
| ----- | --------- |
| Jan   | 950,000   |
| Feb   | 2,100,000 |
| Mar   | 2,200,000 |

---

# أول حاجة:

# Error

يعني:

```text id="h4m8q1"
الحقيقي - التوقع
```

---

# يبقى الأخطاء:

| Month | Error   |
| ----- | ------- |
| Jan   | 50k     |
| Feb   | 100k    |
| Mar   | 800k 😄 |

---

# هنا مارس فيه غلطة كبيرة جدًا.

---

# دلوقتي:

# MAE يعني إيه؟

# Mean Absolute Error

يعني:

# متوسط الخطأ العادي.

---

# نحسب:

MAE = \frac{50k + 100k + 800k}{3}

↓

تقريبًا:

```text id="n7q2m5"
316k
```

---

# يعني:

في المتوسط،
الموديل يغلط حوالي:

```text id="t2m8v4"
316k
```

---

# طيب RMSE بقى؟

هنا الفرق المهم 😄

---

# RMSE يعاقب الغلطات الكبيرة جدًا.

---

# بدل ما يجمع errors مباشرة،

يعمل:

Error^2

---

# يعني:

| Error | Squared     |
| ----- | ----------- |
| 50k   | صغير        |
| 100k  | أكبر        |
| 800k  | ضخم جدًا 😄 |

---

# فـ RMSE يطلع أعلى من MAE.

---

# ليه ده useful؟

عشان يقولك:

```text id="r5m2q7"
هل فيه catastrophic mistakes؟
```

---

# لو الـ RMSE قريب من MAE

يبقى:

# الأخطاء مستقرة.

---

# لو RMSE أعلى بكتير

يبقى:

# فيه شوية شهور الموديل غلط فيها جامد.

---

# وده اللي عندكم 😄

---

# عندكم:

## MAE

```text id="c8m1q4"
41k
```

---

## RMSE

```text id="u3q7n2"
80k
```

---

# يعني إيه؟

معناه:

## أغلب الشهور:

الخطأ صغير.

لكن:

## فيه كام شهر:

الخطأ كبير.

---

# غالبًا:

* holidays
* launches
* spikes

---

# طيب R² بقى؟

ده مختلف تمامًا 😄

---

# R² يعني:

# "الموديل فهم قد إيه من حركة الداتا"

---

# أو:

# "قد إيه التوقعات قريبة من الحقيقة overall"

---

# R² عندكم:

```text id="y4m8p1"
97.6%
```

---

# يعني:

الموديل فهم:

```text id="v7q2m5"
حوالي 97% من patterns المبيعات
```

---

# وده ممتاز جدًا.

---

# طيب ليه نهتم بـ R² أكتر عندكم؟

لأن:
الـ sales عندكم:

# بالملايين.

---

# فـ RMSE:

```text id="p1m7q4"
80k
```

شكله كبير،
لكن مقارنة بـ:

```text id="j5q8m2"
2M
3M
```

هو صغير نسبيًا.

---

# يعني نسبة الخطأ تقريبًا:

\frac{80k}{2M} \approx 4%

---

# وده ممتاز.

---

# الخلاصة البسيطة جدًا

| Metric | معناه                                   |
| ------ | --------------------------------------- |
| MAE    | متوسط الخطأ الطبيعي                     |
| RMSE   | متوسط الخطأ مع عقاب قوي للغلطات الكبيرة |
| R²     | قد إيه الموديل فهم الـ patterns         |

---

# عندكم:

## MAE منخفض → جيد

## RMSE أعلى → فيه few difficult months

## R² عالي جدًا → الموديل overall ممتاز 😄
بالظبط 😄
وده الصح أصلًا في حالتكم.

---

# خلي بالك من الفرق المهم جدًا:

## فيه:

# UI Filters

وفيه:

# Model Features

---

# الـ user هيختار:

* Country
* City
* Store

في الـ Streamlit.

---

# لكن الموديل نفسه:

مش لازم ياخد:

```python id="qz4t21"
country_encoded
city_encoded
```

---

# ليه؟

لأن:

# store_id كفاية جدًا.

---

# لأن كل store أصلًا:

مرتبط بـ:

* city معينة
* country معينة
* behavior معين

---

# يعني لما الموديل يشوف:

```python id="x8an92"
store_encoded = 17
```

هو already فهم:

* الدولة
* المدينة
* طبيعة المبيعات
* seasonality
* القوة الشرائية

---

# فلو ضفت:

```python id="f2mb81"
country_encoded
city_encoded
```

يبقى غالبًا:

# redundancy

وممكن يساعد:

# overfitting

---

# لكن:

هل نمسح store_encoded كمان؟

# لأ 😄

ده مهم جدًا.

---

# لأن الموديل لازم يعرف:

هو بيتوقع لمين أصلًا.

---

# فالنظام الصح:

| حاجة          | موجودة في UI | موجودة في Model |
| ------------- | ------------ | --------------- |
| Country       | ✅            | ❌               |
| City          | ✅            | ❌               |
| Store         | ✅            | indirectly      |
| store_encoded | ❌            | ✅               |

---

# يعني الـ flow:

## user:

يختار:

```text id="nyujvf"
USA
↓
New York
↓
Apple Fifth Avenue
```

---

# التطبيق يحول ده داخليًا لـ:

```python id="z1cw83"
store_encoded = 17
```

---

# وبعدها:

الموديل يشتغل.

---

# وده actually:

أفضل architecture عندكم 😄
