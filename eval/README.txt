Оценка точности анализа.

labels.csv — эталон: решения Skycoach из примеров выплат (banner_review_examples.docx)
и синтетические кейсы, собранные из реальных роликов (make_synthetic.py).
split: calibration — на этих роликах подбирались пороги (оценка на них оптимистична);
       holdout — новые размеченные ролики, НЕ использованные для подбора (добавлять сюда);
       synthetic — искусственные проверки крайних случаев.

  python eval/evaluate.py                # полный пайплайн (нужен NeuroAPI)
  python eval/evaluate.py --logo-only    # без vision-модели: только логотип по эталону
  python eval/evaluate.py --only DbQGs9HMcOQ syn_clipped

Видео кэшируются в eval/cache/ (не в git): Apify и скачивание — только при первом запуске.
