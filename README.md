# Cookie Clicker Bot (YOLO)

Бот для автоматической игры в Cookie Clicker: сам делает скриншоты, размечает
датасет, обучает YOLO-детектор и кликает по нужным элементам игры.

## Файлы проекта

| Файл | Назначение |
|------|-----------|
| [`screener.py`](screener.py) | Делает скриншоты экрана каждые 5 секунд в папку `screenshots/` |
| [`augment.py`](augment.py) | Генерирует аугментированные копии скриншотов (albumentations) |
| [`autolabel.py`](autolabel.py) | Автоматическая разметка изображений моделью |
| [`split_dataset.py`](split_dataset.py) | Раскладывает размеченные данные по `images/` и `labels/` (train/val) |
| [`train.py`](train.py) | Обучение YOLO-модели |
| [`app.py`](app.py) | Запуск бота: детекция экрана и клики по элементам игры |

## Структура датасета

```
images/
├── train/   # 80 % картинок
└── val/     # 20 % картинок
labels/
├── train/   # разметка YOLO (class cx cy w h)
└── val/
data.yaml    # конфиг для обучения
class.txt    # список классов
```

## Порядок работы

1. `python screener.py` — набрать скриншоты игры в `screenshots/`.
2. Разметить их (например, в X-AnyLabeling) — появятся пары `.png` + `.txt`.
3. `python augment.py` — при необходимости размножить данные.
4. `python split_dataset.py` — разложить по `images/` и `labels/` (80/20).
5. `python train.py` — обучить модель.
6. `python app.py` — запустить бота.

## Установка

```bash
pip install -r requirements.txt
```

## Классы

- `game` — игровые объекты
- `upgrade` — улучшения
- `advertising` — реклама (закрывается автоматически)
