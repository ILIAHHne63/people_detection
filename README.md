# Детекция людей на видео с использованием модели YOLO

В данном репозитории представлена демонстрация работы модели компьютерного зрения YOLO для детекции людей на видео. Модель YOLO позволяет эффективно обнаруживать объекты на изображениях и видео в реальном времени.

## Установка зависимостей

Перед началом работы необходимо установить все необходимые зависимости. Рекомендуемая версия Python — 3.10.5.

Для всех платформ можно использовать следующие команды:

```bash
pip install ultralytics
pip install argparse     
```
Также можно установить зависимости из файла requirements.txt:

```bash
pip install -r requirements.txt
```

## Исходное видео

Исходное видео можно загрузить по ссылке:
https://drive.google.com/file/d/1COCqVvT8uOERn-pq9mBZxXEjvrRDD3ix/view?usp=share_link

## Запуск детекции

Для запуска процесса детекции людей на видео используйте следующую команду:


```bash
python3 detect.py --input "crowd.mp4" --output "crowd_detection.mp4" --model "yolo12x.pt"
```

Параметры:


```
--input
```
- Путь к исходному видеофайлу.

```
--output
```
- Путь для сохранения обработанного видео.

```
--model
```
- Путь к файлу весов модели YOLO. По умолчанию 'yolo12x.pt'.

После выполнения кода вы получите видео с отмеченными людьми, обнаруженными моделью YOLO.

## Видео с детекцией

Результат работы модели можно посмотреть по ссылке:
https://drive.google.com/file/d/1pMsOTDafI8QRHjriilzRPRxABXqV5Nx5/view?usp=share_link

## Анализ и улучшение модели


Данная модель демонстрирует высокую скорость работы и соответствует уровню state-of-the-art (SOTA) среди моделей детекции. Она качественно предсказывает bounding boxes, однако имеет ряд ограничений. Например, модель может пропускать объекты из-за их малого размера или перекрытия, а также, в теории, может хуже справляться с задачами в условиях плохого качества видео или недостаточного освещения. Для более детального анализа качества работы модели можно использовать размеченные данные и стандартные метрики, такие как mAP (mean Average Precision), которые позволяют оценить точность и полноту детекции.

Основные проблемы модели:

- Ложные срабатывания:
В некоторых случаях модель может ошибочно детектировать объекты, не относящиеся к людям. Например, на видео отчетливо видно, как модель детектирует картину с изображением человека. Это может быть связано с недостаточным качеством или разнообразием данных, использованных для обучения.
- Пропуски объектов:
В crowded-сценах (например, в толпе) модель может пропускать некоторые объекты из-за их малого размера или перекрытия другими людьми.

Возможные улучшения:

- Дообучение модели:
Для повышения точности можно дообучить модель на специализированном наборе данных, содержащем изображения людей в различных условиях: при плохом освещении, в толпе, с перекрытиями и т.д.
- Постобработка результатов:
Добавление фильтров для удаления ложных срабатываний (например, на основе размера, формы или контекста объектов) может значительно улучшить качество детекции. Хорошим примером такого фильтра может служить AlphaCLIP https://github.com/SunzeY/AlphaCLIP. Данная модель принимает текстовое описание, маску и изображение, а затем оценивает уверенность в соответствии изображения текстовому промпту. В данном случае в качестве текстового промпта можно использовать название класса.
- Оптимизация трекинга:
Использование более сложных алгоритмов трекинга может повысить стабильность отслеживания объектов, особенно в сложных сценах с большим количеством движущихся людей.
- Улучшение входных данных:
Предварительная обработка видео (например, коррекция освещения, стабилизация кадров или повышение разрешения) может положительно сказаться на качестве детекции.

## Итог:

Модель YOLO демонстрирует высокую эффективность в задачах детекции людей, но для достижения наилучших результатов в сложных условиях требуется дополнительная настройка и улучшение. Использование специализированных данных, самых мощных версий модели и оптимизация трекинга помогут повысить точность и стабильность работы модели.
