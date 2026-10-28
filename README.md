# 🍪 Cookie Clicker Bot (YOLO)

**Что это.** Бот, который сам играет в **Cookie Clicker**. Он снимает экран, распознаёт
игровые объекты нейросетью (YOLO) и кликает: по печеньке, по улучшениям и по рекламе,
которую сразу закрывает. Модель можно обучить на своих скриншотах, а можно сразу
использовать готовую [`bestV1.onnx`](bestV1.onnx).

**Что распознаёт** (классы, см. [`class.txt`](class.txt)):

| id | класс | что делает бот |
|----|-------|----------------|
| 0 | `game` | кликает по игровым объектам (основная «печенька» и т.п.) |
| 1 | `upgrade` | раз в 7 секунд проходит по улучшениям снизу вверх, затем слева направо, и покупает их |
| 2 | `advertising` | закрывает рекламные окна сразу, с наивысшим приоритетом |

Этот README — инструкция по **установке и запуску** на Windows, macOS и Linux.

---

## 📁 Файлы проекта

| Файл | Назначение |
|------|-----------|
| [`app.py`](app.py) | **Запуск бота.** Делает скриншоты, гоняет YOLO, кликает по `game`/`upgrade`/`advertising`. Пробел — пауза. Берёт самую свежую модель из `runs/detect/*/weights/`, а если обученных нет — готовую `bestV1.pt`. |
| [`screener.py`](screener.py) | Снимает скриншоты экрана каждые 5 секунд — из них собирается датасет. |
| [`augment.py`](augment.py) | Размножает изображения аугментациями (albumentations), чтобы данных для обучения было больше. |
| [`split_dataset.py`](split_dataset.py) | Раскладывает размеченные пары `jpg`+`txt` по `images/` и `labels/` (train/val) для Ultralytics. |
| [`train.py`](train.py) | **Обучение YOLO-модели** по [`data.yaml`](data.yaml). |
| [`data.yaml`](data.yaml) | Конфиг датасета для обучения: пути к картинкам и список классов. |
| [`config.yaml`](config.yaml) | Конфиг детектора (тип модели, пороги, список классов). |
| [`class.txt`](class.txt) | Список классов в нужном порядке — им руководствуются при разметке. |
| [`bestV1.pt`](bestV1.pt) | **Готовая модель** (ultralytics/torch) — используется по умолчанию, если своих обученных весов нет. Работает сразу после установки `requirements.txt`. |
| [`bestV1.onnx`](bestV1.onnx) | ONNX-версия готовой модели. Нужен `onnxruntime` (не входит в `requirements.txt`), поэтому подключается только как альтернатива. |
| [`requirements.txt`](requirements.txt) | Зависимости Python (`ultralytics`, `pyautogui`, `opencv`, `pynput`, …). |

---

## ✅ Требования

- **Python 3.10 – 3.12** (рекомендуется 3.11). Проверить: `python --version`.
- ~5 ГБ свободного места (PyTorch + веса + датасет).
- Интернет для установки зависимостей и загрузки предобученных весов (`yolov8n.pt`).
- **ОС:** Windows 10/11, macOS 12+, Linux (X11/XWayland).
- *(Опционально)* NVIDIA GPU с CUDA — ускоряет обучение и работу бота.

> ⚠️ Бот использует [`pyautogui`](https://pyautogui.readthedocs.io/): он снимает **весь экран**
> и двигает **реальный курсор**. Игра должна быть на переднем плане, экран не должен
> блокироваться (скринсейвер, lock screen).

---

## 🚀 Установка

Общая идея одна: создать виртуальное окружение → активировать → установить зависимости.

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

   Fedora:

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

5. **Нужна сессия X11.** В **Wayland** `pyautogui` не может снимать экран и управлять
   курсором. В меню входа выберите режим *«Ubuntu on Xorg»* / *X11*.

> ℹ️ Для GPU-ускоренного PyTorch (CUDA) поставьте сборку под свою версию CUDA с
> [pytorch.org](https://pytorch.org/get-started/locally/), затем выполните
> `pip install -r requirements.txt`.

### 🖥️ Проверка установки

Активированное окружение — в начале строки терминала появится `(.venv)`. Быстрая проверка:

```bash
python -c "import pyautogui, cv2, numpy, albumentations, ultralytics, pynput; print('OK')"
```

---

## ▶️ Запуск

Запускайте команды из корня проекта с активированным окружением.

**1. Собрать скриншоты игры** (игра на переднем плане, полноэкранно):

```bash
python screener.py
```

Кадры сохраняются каждые 5 секунд в `C:\cookie_screen` (или путь, заданный в
[`screener.py`](screener.py)). Остановка — `Ctrl+C`. Наберите **100–300** кадров.

**2. Разметить данные.** Разметьте скриншоты в любом YOLO-редакторе
([X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) или
[LabelImg](https://github.com/HumanSignal/labelImg)). Классы и их порядок — как в
[`class.txt`](class.txt): `game`, `upgrade`, `advertising`.

**3. Аугментация (опционально)** — если данных мало:

```bash
python augment.py
```

**4. Разложить по train/val:**

```bash
python split_dataset.py --src itog        # 80/20, создаст images/, labels/ и data.yaml
```

**5. Обучить модель:**

```bash
python train.py
```

Веса сохранятся в `runs/detect/cookie_detector*/weights/best.pt`. Бот сам берёт
**самую свежую** модель из `runs/detect/*/weights/`, так что переобучение подхватится
автоматически.

**6. Запустить бота** (окно игры открыто и видно):

```bash
python app.py
```

### 🎮 Управление ботом

- **Пробел** — пауза / продолжить (в любой момент).
- **Ctrl+C** — полная остановка (иногда не срабатывает из-за захвата ввода — тогда
  закройте терминал).

---

## 🛠️ Решение проблем

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError: No module named 'ultralytics'` | Активировано ли окружение `(.venv)`? Проверьте `pip -V` и `python -c "import sys; print(sys.executable)"`. |
| `OSError: [WinError 10060]` при `train.py` | Нет интернета для `yolov8n.pt` — скачайте вручную и положите в корень проекта. |
| `FileNotFoundError: Не найдена модель для запуска` в `app.py` | Нет ни обученных весов в `runs/detect/*/weights/`, ни `bestV1.pt`/`bestV1.onnx` в корне. Обучите модель (`python train.py`) или положите готовую `bestV1.pt` рядом с [`app.py`](app.py). |
| Скриншот получается чёрным | Linux: Wayland → переключитесь на X11. macOS: не выдано разрешение «Запись экрана». |
| Курсор не двигается / клики не работают | macOS: не выдано «Универсальный доступ». Linux: нужен `python3-xlib`. |
| Кириллица в консоли Windows «кракозябрами» | Выполните `chcp 65001` или используйте Windows Terminal. |
| Обучение очень медленное | GPU-сборка PyTorch или меньше `epochs`/`imgsz` в [`train.py`](train.py). |

---

## 📄 Лицензия

Учебный/личный проект для автоматизации Cookie Clicker. Используйте на свой страх и риск —
автоматизация может противоречить правилам конкретного сайта/игры.
