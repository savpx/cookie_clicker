# 🍪 Cookie Clicker Bot (YOLO)

Бот для автоматической игры в **Cookie Clicker**. Он делает скриншоты экрана, помогает
разметить и размножить датасет, обучает YOLO-детектор и затем сам играет: кликает по
игровым объектам, покупает улучшения и закрывает рекламу.

Поддерживаемые классы (см. [`class.txt`](class.txt)):

| id | класс | что делает бот |
|----|-------|----------------|
| 0 | `game` | кликает по игровым объектам (основная «печенька» и т.п.) |
| 1 | `upgrade` | раз в 10 секунд проходит по всем улучшениям сверху вниз и покупает их |
| 2 | `advertising` | закрывает рекламные окна сразу, с наивысшим приоритетом |

---

## 📁 Файлы проекта

| Файл | Назначение |
|------|-----------|
| [`screener.py`](screener.py) | Делает скриншоты экрана каждые 5 секунд |
| [`augment.py`](augment.py) | Генерирует аугментированные копии изображений (albumentations) |
| [`split_dataset.py`](split_dataset.py) | Раскладывает размеченные данные по `images/` и `labels/` (train/val) |
| [`train.py`](train.py) | Обучение YOLO-модели |
| [`app.py`](app.py) | Запуск бота: детекция экрана и клики по элементам игры |
| [`data.yaml`](data.yaml) | Конфиг датасета для обучения |
| [`class.txt`](class.txt) | Список классов |
| [`requirements.txt`](requirements.txt) | Зависимости Python |

---

## ✅ Требования

- **Python 3.10 – 3.12** (рекомендуется 3.11). Проверить версию: `python --version`.
- ~5 ГБ свободного места (PyTorch + веса модели + датасет).
- Интернет для установки зависимостей и загрузки предобученных весов (`yolov8n.pt`).
- **ОС:** Windows 10/11, macOS 12+, Linux (X11/XWayland).
- *(Опционально)* NVIDIA GPU с CUDA — заметно ускоряет обучение и работу бота.

> ⚠️ **Важно про скриншоты и клики.** Бот использует [`pyautogui`](https://pyautogui.readthedocs.io/) —
> он снимает **весь экран** и двигает **реальный курсор**. Игра должна быть запущена
> на переднем плане, а экран не должен блокироваться (скринсейвер, lock screen).
> Закрывайте окна, которые мешают обзору игрового поля.

---

## 🚀 Установка

Ниже — пошагово для каждой ОС. Общая идея одинаковая:
создать виртуальное окружение → активировать → установить зависимости.

### 🪟 Windows (cmd / PowerShell)

1. Установите Python с [python.org](https://www.python.org/downloads/) и **обязательно**
   отметьте галочку **«Add python.exe to PATH»**.
2. Клонируйте репозиторий и перейдите в папку проекта:

   ```bat
   git clone https://github.com/savpx/cookie.git
   cd cookie
   ```

3. Создайте и активируйте виртуальное окружение:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   Для PowerShell, если ругается на политику запуска скриптов:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. Обновите `pip` и установите зависимости:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 🍎 macOS

1. Установите Python 3.11 (через [Homebrew](https://brew.sh/)):

   ```bash
   brew install python@3.11 git
   ```

2. Клонируйте репозиторий и перейдите в папку проекта:

   ```bash
   git clone https://github.com/savpx/cookie.git
   cd cookie
   ```

3. Создайте и активируйте виртуальное окружение:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. Установите зависимости:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Выдайте разрешения** (macOS блокирует управление экраном по умолчанию):
   - *Системные настройки → Конфиденциальность и безопасность → **Запись экрана*** →
     добавьте Terminal (или iTerm).
   - *Системные настройки → Конфиденциальность и безопасность → **Универсальный доступ*** →
     добавьте Terminal (или iTerm), чтобы бот мог двигать курсор.
   - После выдачи разрешений **перезапустите терминал**.

### 🐧 Linux (Debian / Ubuntu / Mint)

1. Установите системные зависимости (нужны для скриншотов и GUI-утилит `pyautogui`):

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip python3-tk python3-dev \
                       python3-xlib scrot git
   ```

   Дистрибутивы на базе Fedora:

   ```bash
   sudo dnf install -y python3 python3-pip python3-tkinter scrot git
   ```

   Arch Linux:

   ```bash
   sudo pacman -S --needed python python-pip tk scrot git
   ```

2. Клонируйте репозиторий и перейдите в папку проекта:

   ```bash
   git clone https://github.com/savpx/cookie.git
   cd cookie
   ```

3. Создайте и активируйте виртуальное окружение:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Установите зависимости:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Сессия X11 обязательна.** В сессии **Wayland** `pyautogui` не может снимать экран
   и управлять курсором. Выберите в меню входа режим *«Ubuntu on Xorg»* / *X11*.

> ℹ️ Если нужен GPU-ускоренный PyTorch (обучение/инференс на CUDA), поставьте сборку
> под свою версию CUDA по инструкции с [pytorch.org](https://pytorch.org/get-started/locally/),
> а затем выполните `pip install -r requirements.txt`.

### 🖥️ Проверка установки

Активированное окружение — в начале строки терминала появится `(.venv)`.
Быстрая проверка, что всё видно и кликается:

```bash
python -c "import pyautogui, cv2, numpy, albumentations, ultralytics; print('OK')"
```

---

## 🕹️ Порядок работы (полный пайплайн)

<details>
<summary><b>Шаг 0. Настройка путей (Windows-пути по умолчанию)</b></summary>

В [`screener.py`](screener.py) и [`augment.py`](augment.py) путь к папке со скриншотами
задан в Windows-стиле:

```python
save_dir = r"C:\cookie_screen"   # screener.py
INPUT_DIR = r"C:\cookie_screen"  # augment.py
OUTPUT_DIR = r"C:\cookie_screen" # augment.py
```

**Пользователям macOS/Linux** нужно заменить их на путь своей ОС, например:

```python
save_dir = "/Users/ваш_логин/cookie_screen"   # macOS
save_dir = "/home/ваш_логин/cookie_screen"    # Linux
```

</details>

### Шаг 1. Собрать скриншоты игры

Запустите игру в полноэкранном режиме (или окном на весь экран), затем:

```bash
python screener.py
```

Скриншоты сохраняются каждые 5 секунд в папку `C:\cookie_screen` (или путь, который вы
задали). Остановка — `Ctrl+C`. Наберите **100–300** кадров, покрывающих разные состояния игры.

### Шаг 2. Разметить датасет

Разметьте скриншоты в любом YOLO-совместимом редакторе, например
[**X-AnyLabeling**](https://github.com/CVHub520/X-AnyLabeling) или
[LabelImg](https://github.com/HumanSignal/labelImg).

Классы и их порядок должны совпадать с [`class.txt`](class.txt):
`game`, `upgrade`, `advertising`.

Рядом с каждым `.png` появится одноимённый `.txt` — разметка в формате YOLO
(`class cx cy w h` в нормализованных координатах).

### Шаг 3. Аугментация (опционально)

Если данных мало — размножьте их (по умолчанию ×5 копий на изображение):

```bash
python augment.py
```

> ⚠️ `augment.py` **не переносит разметку** — она только для изображений. Размечайте
> после аугментации или используйте другой инструмент, генерирующий пары `.jpg` + `.txt`.

### Шаг 4. Разложить по train/val

[`split_dataset.py`](split_dataset.py) берёт пары `<имя>.jpg` + `<имя>.txt` из папки-источника
(по умолчанию `itog`) и раскладывает их в структуру Ultralytics. Аугментации одной исходной
картинки **не разделяются** между выборками (защита от data leakage).

```bash
# сначала посмотреть план, ничего не меняя
python split_dataset.py --src itog --dry-run

# реальный запуск: 80 / 20, создаст images/, labels/ и data.yaml
python split_dataset.py --src itog

# свои проценты + перенос файлов вместо копирования
python split_dataset.py --src itog --train 70 --val 20 --test 10 --move
```

Полезные флаги: `--no-group` (делить по отдельным картинкам), `--seed` (воспроизводимость),
`--no-data-yaml` (не перезаписывать конфиг).

### Шаг 5. Обучить модель

```bash
python train.py
```

Скрипт берёт `yolov8n.pt` (скачается автоматически) и обучает детектор по [`data.yaml`](data.yaml):
100 эпох, `imgsz=640`, `batch=16`, ранняя остановка при 50 эпохах без улучшений.

Результаты появятся в `runs/detect/cookie_detector/`. Нужные веса:
`runs/detect/cookie_detector/weights/best.pt`.

### Шаг 6. Запустить бота

Убедитесь, что окно игры открыто и видно на экране, затем:

```bash
python app.py
```

Бот зациклится: делает скриншот → прогоняет YOLO → кликает. Остановка — `Ctrl+C` (работает
не всегда из-за захвата ввода — в таком случае закройте терминал).

Поведение задаётся константами в [`app.py`](app.py):

| Константа | По умолчанию | Смысл |
|-----------|--------------|-------|
| `CONF_THRESHOLD` | `0.7` | Порог уверенности детекции |
| `UPGRADE_COOLDOWN` | `10.0` | Пауза между проходами по улучшениям (сек) |

---

## 🗂️ Структура датасета

```
images/
├── train/   # ~80 % изображений
└── val/     # ~20 % изображений
labels/
├── train/   # разметка YOLO (class cx cy w h)
└── val/
data.yaml    # конфиг для обучения
class.txt    # список классов
```

`data.yaml`:

```yaml
path: ./
train: images/train
val: images/val
nc: 3
names: ['game', 'upgrade', 'advertising']
```

---

## 🛠️ Решение проблем

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'ultralytics'` | Активировано ли окружение `(.venv)`? Проверьте `pip -V` и путь `python -c "import sys; print(sys.executable)"`. |
| `OSError: [WinError 10060]` при `train.py` | Нет интернета для скачивания `yolov8n.pt` — скачайте вручную и положите в корень проекта. |
| `FileNotFoundError: ... best.pt` в `app.py` | Модель ещё не обучена — выполните шаги 4–5, либо укажите свой путь в [`app.py`](app.py:8). |
| Скриншот получается чёрным | Linux: сессия Wayland — переключитесь на X11. macOS: не выдано разрешение «Запись экрана». |
| Курсор не двигается / клики не работают | macOS: не выдано «Универсальный доступ». Linux: нужен `python3-xlib`. |
| `IndexError` / пустой результат в `augment.py` | `INPUT_DIR` пустой либо путь недоступен — проверьте папку со скриншотами. |
| Кириллица в консоли Windows «кракозябрами» | Выполните `chcp 65001` перед запуском, либо используйте Windows Terminal. |
| Обучение очень медленное | Используйте GPU-сборку PyTorch, уменьшите `epochs`/`imgsz` в [`train.py`](train.py). |

---

## 📄 Лицензия

Учебный/личный проект для автоматизации Cookie Clicker. Используйте на свой страх и риск —
автоматизация может противоречить правилам конкретного сайта/игры.
