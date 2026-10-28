"""Cookie Clicker Bot — детекция экрана через YOLO и автоматические клики.

Логика работы цикла:
  1. Реклама      — закрывается сразу, как только появилась (высший приоритет).
                    Кликаем по УГЛУ рамки (там крестик, а не в центре — центр
                    это сама реклама). После клика результат перепроверяется
                    несколько раз, при промахе пробуется следующий угол.
  2. Улучшения    — раз в UPGRADE_INTERVAL секунд проход СНИЗУ ВВЕРХ, затем
                    СЛЕВА НАПРАВО. Перед каждым кликом детекция обновляется,
                    чтобы координаты были актуальны и ничего не пропускалось.
  3. Печенька     — кликается постоянно, без паузы, максимально быстро.
"""

import glob
import os
import time
from collections import namedtuple

import cv2
import numpy as np
import pyautogui
from pynput import keyboard as pynput_keyboard
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Скорость
# ---------------------------------------------------------------------------
# pyautogui по умолчанию делает паузу 0.1 с ПОСЛЕ каждого действия (клика,
# перемещения и т.п.). Обнуляем её, чтобы клики шли подряд без задержек.
pyautogui.PAUSE = 0

# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
# Готовые модели в корне проекта — используются, если обученных весов нет.
FALLBACK_MODELS = ["bestV1.pt", "bestV1.onnx"]


def find_latest_weights():
    """Найти самую свежую обученную модель в runs/detect/*/weights/.

    YOLO при каждом обучении создаёт новую папку (cookie_detector,
    cookie_detector-2, ...), поэтому вместо жёсткого пути берём самую
    недавно обученную модель. Сначала ищем best.pt, затем last.pt
    (last.pt сохраняется, даже если best не обновился).
    Возвращает путь или None, если обученных весов нет.
    """
    for pattern in ("best.pt", "last.pt", "*.pt"):
        candidates = glob.glob(
            os.path.join("runs", "detect", "*", "weights", pattern)
        )
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def resolve_model_path():
    """Выбрать модель: явное указание, свежая обученная, иначе готовая из корня.

    Приоритет:
      1. env COOKIE_MODEL  — можно принудительно вернуться на старую модель,
                             например: set COOKIE_MODEL=bestV1.pt
      2. самая свежая обученная в runs/detect/*/weights/ (best.pt/last.pt)
      3. готовая модель из корня проекта (bestV1.pt / bestV1.onnx)

    После переобучения шаг 2 всегда подхватывает НОВУЮ модель. Если она хуже
    детектит улучшения — переменной COOKIE_MODEL можно быстро откатиться.
    """
    forced = os.environ.get("COOKIE_MODEL")
    if forced:
        if not os.path.exists(forced):
            print(f"[warn] COOKIE_MODEL='{forced}' не найден — ищу автоматически.")
        else:
            return forced

    trained = find_latest_weights()
    if trained:
        return trained
    for fallback in FALLBACK_MODELS:
        if os.path.exists(fallback):
            return fallback
    return FALLBACK_MODELS[0]


MODEL_PATH = resolve_model_path()
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Не найдена модель для запуска.\n"
        "  • Обучите её: python train.py (веса появятся в runs/detect/*/weights/)\n"
        "  • Либо положите готовую bestV1.pt (или bestV1.onnx) в корень проекта.\n"
        f"  Искал здесь: {MODEL_PATH}"
    )
print(f"Загружаю модель: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ---------------------------------------------------------------------------
# Индексы классов (совпадают с data.yaml / class.txt)
# ---------------------------------------------------------------------------
CLASS_GAME = 0         # game
CLASS_UPGRADE = 1      # upgrade
CLASS_ADVERTISING = 2  # advertising

# ---------------------------------------------------------------------------
# Пороги уверенности (можно задать свой для каждого класса)
# ---------------------------------------------------------------------------
# Улучшения — мелкие иконки, детектятся хуже, поэтому порог ниже: иначе
# «Курсор» и подобные кнопки просто отфильтровываются и не покупаются.
#
# ВАЖНО: если после переобучения модель находит как «улучшение» только один
# объект — порог тут не поможет, дело в весах. Смотри debug_frame.jpg (DEBUG_SAVE_FRAME):
# он покажет, что модель реально видит. Быстрый откат — set COOKIE_MODEL=bestV1.pt.
# Порог для upgrade можно менять на лету: set COOKIE_UPGRADE_CONF=0.35
CONF_THRESHOLDS = {
    CLASS_GAME: 0.55,
    CLASS_UPGRADE: 0.40,
    CLASS_ADVERTISING: 0.50,
}
CONF_THRESHOLD = 0.50  # запасной порог для неизвестных классов

# Позволить задать порог для улучшений через переменную окружения (без правки кода).
_upgrade_conf_env = os.environ.get("COOKIE_UPGRADE_CONF")
if _upgrade_conf_env:
    try:
        CONF_THRESHOLDS[CLASS_UPGRADE] = float(_upgrade_conf_env)
        print(f"Порог для улучшений переопределён: {CONF_THRESHOLDS[CLASS_UPGRADE]}")
    except ValueError:
        print(f"[warn] COOKIE_UPGRADE_CONF='{_upgrade_conf_env}' не число — игнорирую.")

# Имена и цвета классов — для отладочной картинки с рамками.
CLASS_NAMES = {CLASS_GAME: "game", CLASS_UPGRADE: "upgrade", CLASS_ADVERTISING: "advertising"}
CLASS_COLORS = {CLASS_GAME: (0, 255, 0), CLASS_UPGRADE: (255, 128, 0), CLASS_ADVERTISING: (0, 0, 255)}

# ---------------------------------------------------------------------------
# Настройки поведения
# ---------------------------------------------------------------------------
UPGRADE_INTERVAL = 15.0  # полный проход по улучшениям — раз в 15 секунд
UPGRADE_CLICKS = 2       # кликов по каждому улучшению (чтобы покупка точно прошла)
UPGRADE_MAX_STEPS = 20   # защита от зацикливания во время прохода
# Пауза после клика: купленное улучшение исчезает/меняется не мгновенно, поэтому
# без неё следующая детекция возвращает ту же иконку и бот кликает впустую.
UPGRADE_SETTLE = 0.18
# Радиус (px) сопоставления «это то же улучшение, что уже обрабатывали».
# Детекция между кадрами дрожит на несколько пикселей, поэтому точного совпадения
# координат мало — иначе одно улучшение обрабатывается по кругу.
UPGRADE_MATCH_RADIUS = 22
CLICKS_PER_BURST = 25    # сколько быстрых кликов по печеньке до новых скриншотов

# --- Обходной путь для «панели одним боксом» ---
# После переобучения модель иногда отдаёт ОДНУ высокую рамку `upgrade` на всю
# панель улучшений. Клик по её центру всегда попадает в середину списка — поэтому
# бот «кликает только по Бабуле». Такие крупные рамки разбиваем на строки:
#   • выше UPGRADE_BIG_BOX_MIN_HEIGHT — считаем рамкой-панелью и режем на строки;
#   • UPGRADE_PANEL_HEADER — «шапка» панели (ряд мелких иконок + «Купить/Продать»),
#     сверху столько пикселей пропускаем;
#   • UPGRADE_ROW_HEIGHT — примерная высота одной строки апгрейда;
#   • UPGRADE_COLUMN_OFFSET — отступ от левого края панели до центра иконки покупки.
# Значения подобраны под текущую вёрстку Cookie Clicker Old. Если промахивается —
# покрути их (или вообще отключи разбиение: UPGRADE_BIG_BOX_MIN_HEIGHT = 100000).
UPGRADE_BIG_BOX_MIN_HEIGHT = 220
UPGRADE_PANEL_HEADER = 130
# Высота строки взята из разметки датасета (itog/*.txt): у строк-улучшений
# h≈0.057 от высоты экрана, шаг между строками ≈0.065. Для 1080p это ≈62 px
# и ≈70 px соответственно — поэтому 70.
UPGRADE_ROW_HEIGHT = 70
UPGRADE_COLUMN_OFFSET = 70
# Верхний ряд панели — мелкие иконки («шапка»). Их тоже надо проходить слева
# направо, поэтому генерируем колонки по всей ширине панели. Значения из
# разметки: центры иконок шапки cx≈0.659 и 0.693 (шаг ≈0.034·1920≈65 px),
# первая иконка ≈на 30 px правее левого края панели.
UPGRADE_HEADER_ICON_OFFSET = 45   # центр ряда иконок шапки от верха панели
UPGRADE_HEADER_MARGIN_X = 30      # отступ до центра первой иконки шапки
UPGRADE_COLUMN_WIDTH = 64         # шаг между иконками шапки слева направо
UPGRADE_HEADER_ICON_HALF = 28     # половина высоты иконки шапки (для клипа)

# --- Служебные зоны Яндекс Игр, куда кликать НЕЛЬЗЯ ---
# Справа на странице висит реклама, снизу — панель с иконками других игр. Модель
# иногда принимает их за «улучшение»/«печеньку», и клик по ним открывает чужую
# игру вместо покупки. Детекции в этих полосах отбрасываем. Значения — доли от
# размера экрана, поэтому работают при любом разрешении. Если будешь играть в
# полноэкранном режиме без боковых панелей (F11) — поставь оба значения в 0.0.
IGNORE_BOTTOM_FRACTION = 0.10
IGNORE_RIGHT_FRACTION = 0.18

# --- Реклама ---
AD_MAX_ATTEMPTS = 6      # максимум одиночных кликов по одной рекламе
AD_ATTEMPT_DELAY = 0.20  # пауза между кликами (чтобы интерфейс успел отреагировать)
AD_SETTLE = 0.30         # пауза после закрытия рекламы перед другими действиями

# Проверка результата: сколько раз и с какой паузой перепроверять, что реклама
# действительно исчезла. Нужно, чтобы не долбить по кнопке, пока окно ещё плавно
# закрывается, и не принять «ещё видно» за «не закрылось».
AD_VERIFY_TRIES = 3      # перепроверок подряд, прежде чем кликнуть ещё раз
AD_VERIFY_DELAY = 0.12   # пауза между перепроверками

# Крестик закрытия почти всегда в УГЛУ рамки, а не в центре: клик по центру —
# это сама реклама, он открывает ссылку. Отступ точки клика от угла рамки
# задаётся в пикселях; перебираем несколько значений, потому что крестики
# бывают разного размера, и несколько углов — на случай нестандартной вёрстки.
AD_CORNER_OFFSETS = (20, 34, 50)

# Диагностика: раз в секунду печатать, сколько объектов каждого класса найдено.
DEBUG = True

# Сохранять кадр с нарисованными рамками (debug_frame.jpg) — самый быстрый способ
# понять, что модель на самом деле «видит» после переобучения. Если на картинке
# рамка «upgrade» стоит только на одной иконке — проблема в весах, а не в логике.
DEBUG_SAVE_FRAME = True
DEBUG_FRAME_PATH = "debug_frame.jpg"
DEBUG_FRAME_INTERVAL = 3.0  # как часто перезаписывать кадр (секунды)
_last_frame_save = 0.0

# Детекция: класс + центр + уверенность + рамка
Detection = namedtuple("Detection", "cls cx cy conf x1 y1 x2 y2")

# Время последнего полного прохода по улучшениям
last_upgrade_pass = 0.0

# Флаг паузы: переключается нажатием пробела
paused = False

# Для ограничения частоты отладочных сообщений
_last_debug_log = 0.0

# Кэш размера экрана: pyautogui.size() не бесплатный, а вызывать его пришлось бы
# на каждую детекцию каждого кадра.
_screen_size = None


def screen_size():
    """Размер экрана (width, height); считается один раз и кэшируется."""
    global _screen_size
    if _screen_size is None:
        _screen_size = pyautogui.size()
    return _screen_size


def in_ignored_zone(cx, cy):
    """True, если точка лежит в служебной полосе сайта (реклама справа / панель игр снизу).

    В этих зонах нет настоящих объектов Cookie Clicker, зато модель регулярно
    находит там ложные «улучшения», клик по которым открывает чужие игры.
    """
    screen_w, screen_h = screen_size()
    if IGNORE_RIGHT_FRACTION > 0 and cx > screen_w * (1 - IGNORE_RIGHT_FRACTION):
        return True
    if IGNORE_BOTTOM_FRACTION > 0 and cy > screen_h * (1 - IGNORE_BOTTOM_FRACTION):
        return True
    return False


def save_debug_frame(frame, detections):
    """Сохранить кадр с нарисованными рамками — визуальная диагностика детекции.

    Рисует поверх скриншота все найденные объекты с классом и уверенностью, а
    также контуры служебных зон (реклама справа / панель игр снизу). Перезаписывает
    файл не чаще DEBUG_FRAME_INTERVAL секунд, чтобы не тормозить цикл.
    """
    global _last_frame_save
    now = time.time()
    if now - _last_frame_save < DEBUG_FRAME_INTERVAL:
        return
    _last_frame_save = now

    canvas = frame.copy()
    screen_w, screen_h = screen_size()

    # Контуры игнорируемых зон — сразу видно, что именно отсекается.
    if IGNORE_RIGHT_FRACTION > 0:
        x = int(screen_w * (1 - IGNORE_RIGHT_FRACTION))
        cv2.rectangle(canvas, (x, 0), (screen_w, screen_h), (128, 128, 128), 2)
    if IGNORE_BOTTOM_FRACTION > 0:
        y = int(screen_h * (1 - IGNORE_BOTTOM_FRACTION))
        cv2.rectangle(canvas, (0, y), (screen_w, screen_h), (128, 128, 128), 2)

    for det in detections:
        color = CLASS_COLORS.get(det.cls, (255, 255, 255))
        cv2.rectangle(
            canvas,
            (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)),
            color, 2,
        )
        label = f"{CLASS_NAMES.get(det.cls, det.cls)} {det.conf:.2f}"
        cv2.putText(
            canvas, label, (int(det.x1), max(int(det.y1) - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    # Скриншот в RGB, а cv2 пишет в BGR — конвертируем обратно.
    cv2.imwrite(DEBUG_FRAME_PATH, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def get_detections():
    """Сделать скриншот экрана и вернуть список детекций (list[Detection])."""
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLDS.get(cls, CONF_THRESHOLD):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # Отбрасываем служебные зоны сайта (реклама справа, панель игр снизу):
            # клик по ним открывает чужие игры, а не покупает улучшение.
            if in_ignored_zone(cx, cy):
                continue
            detections.append(
                Detection(
                    cls=cls,
                    cx=cx,
                    cy=cy,
                    conf=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )

    if DEBUG and DEBUG_SAVE_FRAME:
        # screenshot_np сейчас в BGR (после cvtColor выше) — для рисования/сохранения
        # как раз то, что нужно cv2; save_debug_frame конвертирует обратно.
        save_debug_frame(screenshot_np, detections)

    return detections


def log_debug(detections):
    """Не чаще раза в секунду показать, сколько объектов найдено по классам.

    Отдельно печатает список найденных улучшений (координаты + уверенность).
    Это ключевая диагностика: если в списке одно улучшение — значит модель
    «видит» только его, и бот физически не может купить другие (проблема в
    весах модели, а не в логике кликов).
    """
    global _last_debug_log
    now = time.time()
    if now - _last_debug_log < 1.0:
        return
    _last_debug_log = now

    counts = {CLASS_GAME: 0, CLASS_UPGRADE: 0, CLASS_ADVERTISING: 0}
    for det in detections:
        counts[det.cls] = counts.get(det.cls, 0) + 1
    print(
        f"[detect] game={counts[CLASS_GAME]} "
        f"upgrade={counts[CLASS_UPGRADE]} ad={counts[CLASS_ADVERTISING]}"
    )
    upgrades = [d for d in detections if d.cls == CLASS_UPGRADE]
    if upgrades:
        spots = ", ".join(
            f"({d.cx:.0f},{d.cy:.0f}) {d.conf:.2f}"
            for d in sorted(upgrades, key=lambda d: (-d.cy, d.cx))
        )
        print(f"[upgrades] {len(upgrades)} шт: {spots}")


def toggle_pause():
    """Переключить паузу бота и сообщить о новом состоянии."""
    global paused
    paused = not paused
    if paused:
        print("Пауза активна — нажмите пробел, чтобы продолжить.")
    else:
        print("Продолжаю работу.")


def on_key_press(key):
    """Обработчик нажатий клавиатуры: пробел ставит/снимает паузу."""
    if key == pynput_keyboard.Key.space:
        toggle_pause()


def start_pause_listener():
    """Запустить фоновый слушатель клавиатуры (пробел = пауза/продолжить)."""
    listener = pynput_keyboard.Listener(on_press=on_key_press)
    listener.daemon = True
    listener.start()
    return listener


def wait_if_paused():
    """Заблокировать выполнение, пока бот на паузе."""
    while paused:
        time.sleep(0.05)


def click_at(x, y, times=1, delay=0.0):
    """Кликнуть в точку указанное число раз (по умолчанию — без задержек)."""
    for _ in range(times):
        wait_if_paused()
        pyautogui.click(x, y)
        if delay:
            time.sleep(delay)


def _ad_boxes(detections):
    """Все детекции класса «реклама» из готового списка."""
    return [d for d in detections if d.cls == CLASS_ADVERTISING]


def _ad_click_points(x1, y1, x2, y2):
    """Точки-кандидаты внутри рамки рекламы, где обычно лежит крестик.

    Крестик закрытия почти всегда стоит в углу рамки, поэтому перебираем углы
    с фиксированным отступом в пикселях (крестики бывают разного размера).
    """
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    points = []
    seen = set()

    def add(px, py):
        # Держим точку строго внутри рамки (иначе клик уйдёт за пределы окна).
        px = min(max(px, x1 + 2), x2 - 2)
        py = min(max(py, y1 + 2), y2 - 2)
        key = (round(px / 3), round(py / 3))
        if key not in seen:
            seen.add(key)
            points.append((px, py))

    # Правый верхний угол — самый частый вариант, пробуем его первым.
    for dx in AD_CORNER_OFFSETS:
        add(x2 - dx, y1 + dx)
    # Затем остальные углы.
    for dx in AD_CORNER_OFFSETS:
        add(x1 + dx, y1 + dx)
    for dx in AD_CORNER_OFFSETS:
        add(x2 - dx, y2 - dx)
    for dx in AD_CORNER_OFFSETS:
        add(x1 + dx, y2 - dx)

    # Запасной вариант: правый верх с отступом в долях рамки (мелкие баннеры).
    add(x2 - w * 0.04, y1 + h * 0.04)
    return points


def _ad_gone(box):
    """Убедиться, что конкретная рамка рекламы (box) действительно исчезла.

    Делает несколько перепроверок подряд. Реклама считается закрытой, только
    если ни одна детекция больше не перекрывает прежнюю рамку, — так чужие
    ложные срабатывания модели в другом месте экрана не мешают.
    """
    x1, y1, x2, y2 = box
    for _ in range(AD_VERIFY_TRIES):
        time.sleep(AD_VERIFY_DELAY)
        overlap = any(
            (x1 - 20 <= det.cx <= x2 + 20) and (y1 - 20 <= det.cy <= y2 + 20)
            for det in _ad_boxes(get_detections())
        )
        if not overlap:
            return True
    return False


def close_ads(detections):
    """Закрыть рекламу стабильно: клик по УГЛУ рамки -> проверка -> следующий угол.

    Крестик закрытия стоит в углу рамки `advertising`, а не в центре — клик по
    центру попадает в саму рекламу и открывает ссылку. Поэтому перебираем углы
    рамки с разными отступами и после каждого клика перепроверяем результат.

    Возвращает True, если реклама была и её удалось закрыть.
    """
    if not _ad_boxes(detections):
        return False

    tried = set()
    for attempt in range(1, AD_MAX_ATTEMPTS + 1):
        wait_if_paused()

        # Свежая детекция перед каждым кликом: рамка могла измениться.
        ads = _ad_boxes(get_detections())
        if not ads:
            return True  # рекламы уже нет — считаем закрытой

        ad = max(ads, key=lambda d: d.conf)
        box = (ad.x1, ad.y1, ad.x2, ad.y2)

        # Первый ещё не испробованный кандидат; если все кончились — идём по кругу.
        points = _ad_click_points(*box)
        target = None
        for pt in points:
            key = (round(pt[0] / 3), round(pt[1] / 3))
            if key not in tried:
                tried.add(key)
                target = pt
                break
        if target is None:
            target = points[(attempt - 1) % len(points)]

        tx, ty = target
        pyautogui.moveTo(tx, ty)   # hover: некоторые крестики видны только при наведении
        time.sleep(0.05)
        pyautogui.click(tx, ty)
        print(f"Реклама: клик #{attempt} по ({tx:.0f}, {ty:.0f}) conf={ad.conf:.2f}")

        if _ad_gone(box):
            print(f"Реклама закрыта за {attempt} клик(ов).")
            return True

        time.sleep(AD_ATTEMPT_DELAY)

    print(f"Реклама не поддалась за {AD_MAX_ATTEMPTS} попыток — пропускаю до следующего кадра.")
    return False


def _already_clicked(cx, cy, points, radius=UPGRADE_MATCH_RADIUS):
    """True, если рядом с точкой (cx, cy) уже был клик в этом проходе.

    Сравниваем по радиусу, а не по точным координатам: детекция одной и той же
    иконки между кадрами смещается на несколько пикселей.
    """
    return any(
        abs(cx - px) <= radius and abs(cy - py) <= radius
        for px, py in points
    )


def _expand_upgrades(upgrades):
    """Развернуть «панель одним боксом» в отдельные строки-цели.

    Если модель вернула крупную рамку `upgrade` (всю панель целиком), её центр —
    это середина списка, и бот кликает по одному и тому же улучшению. Такую рамку
    режем на строки фиксированной высоты и целимся в левую колонку строки (иконку
    покупки). Мелкие нормальные детекции возвращаются без изменений.
    """
    expanded = []
    for det in upgrades:
        if (det.y2 - det.y1) < UPGRADE_BIG_BOX_MIN_HEIGHT:
            expanded.append(det)  # обычная отдельная иконка — оставляем как есть
            continue

        # 1) «Шапка» панели — ряд мелких иконок сверху. Проходим слева направо
        #    по всей ширине панели.
        header_cy = det.y1 + UPGRADE_HEADER_ICON_OFFSET
        col_x = det.x1 + UPGRADE_HEADER_MARGIN_X
        while col_x <= det.x2 - UPGRADE_HEADER_MARGIN_X:
            expanded.append(
                Detection(
                    cls=det.cls,
                    cx=col_x,
                    cy=header_cy,
                    conf=det.conf,
                    x1=col_x - UPGRADE_HEADER_ICON_HALF,
                    y1=header_cy - UPGRADE_HEADER_ICON_HALF,
                    x2=col_x + UPGRADE_HEADER_ICON_HALF,
                    y2=header_cy + UPGRADE_HEADER_ICON_HALF,
                )
            )
            col_x += UPGRADE_COLUMN_WIDTH

        # 2) Строки-улучшения под шапкой — слева вниз, клик по левой колонке.
        target_x = det.x1 + UPGRADE_COLUMN_OFFSET
        y = det.y1 + UPGRADE_PANEL_HEADER
        while y + UPGRADE_ROW_HEIGHT <= det.y2:
            expanded.append(
                Detection(
                    cls=det.cls,
                    cx=target_x,
                    cy=y + UPGRADE_ROW_HEIGHT / 2,
                    conf=det.conf,
                    x1=det.x1, y1=y, x2=det.x2, y2=y + UPGRADE_ROW_HEIGHT,
                )
            )
            y += UPGRADE_ROW_HEIGHT

    # Снизу вверх, внутри ряда — слева направо (то же правило, что и раньше).
    expanded.sort(key=lambda d: (-d.cy, d.cx))
    return expanded


def buy_all_upgrades():
    """Проход по улучшениям: СНИЗУ ВВЕРХ, затем СЛЕВА НАПРАВО.

    Детекции обновляются ПЕРЕД КАЖДЫМ кликом — координаты всегда актуальны,
    поэтому улучшение (например, «Курсор») не пропускается из-за устаревшего
    кадра. Каждое улучшение обрабатывается ровно один раз за проход (по радиусу,
    чтобы дрожание координат не считалось новым улучшением). Возвращает True,
    если что-то было куплено.
    """
    bought = False
    clicked = []

    for _ in range(UPGRADE_MAX_STEPS):
        wait_if_paused()
        raw_upgrades = [d for d in get_detections() if d.cls == CLASS_UPGRADE]
        if not raw_upgrades:
            break

        # Если модель отдала всю панель одним боксом — режем её на строки.
        # Порядок (снизу вверх, внутри ряда слева направо) задаёт _expand_upgrades.
        upgrades = _expand_upgrades(raw_upgrades)

        # Берём ближайшее улучшение, которое ещё не обрабатывали в этом проходе.
        target = None
        for det in upgrades:
            if not _already_clicked(det.cx, det.cy, clicked):
                target = det
                break

        if target is None:
            # Все найденные улучшения уже обработаны (например, не хватает денег).
            break

        clicked.append((target.cx, target.cy))
        click_at(target.cx, target.cy, times=UPGRADE_CLICKS, delay=0.05)
        print(f"Купил улучшение: ({target.cx:.0f}, {target.cy:.0f}) conf={target.conf:.2f}")
        bought = True

        # Даём панели обновиться: купленное улучшение исчезает/меняется. Без этой
        # паузы следующая детекция может вернуть ту же иконку и клик уйдёт впустую.
        time.sleep(UPGRADE_SETTLE)

    return bought


def main_loop():
    global last_upgrade_pass

    while True:
        # Ждём снятия паузы, если бот остановлен пробелом.
        wait_if_paused()

        detections = get_detections()
        if DEBUG:
            log_debug(detections)

        # 1. Реклама — высший приоритет. Закрываем аккуратно и с проверкой.
        if close_ads(detections):
            # Даём интерфейсу успокоиться и сразу пересканируем экран,
            # чтобы не кликать вслепую по изменившейся картинке.
            time.sleep(AD_SETTLE)
            continue

        # 2. Улучшения — полный проход раз в UPGRADE_INTERVAL секунд.
        now = time.time()
        if now - last_upgrade_pass >= UPGRADE_INTERVAL:
            buy_all_upgrades()
            last_upgrade_pass = now

        # 3. Печенька — кликаем без остановок, максимально быстро.
        cookies = [(d.cx, d.cy) for d in detections if d.cls == CLASS_GAME]
        if not cookies:
            # Ничего не нашли — сразу новый скриншот, без сна.
            continue

        # Быстрая серия кликов, затем снова скриншот (чтобы отслеживать
        # движущиеся объекты вроде золотой печеньки).
        for _ in range(CLICKS_PER_BURST):
            for cx, cy in cookies:
                wait_if_paused()
                pyautogui.click(cx, cy)


if __name__ == "__main__":
    print("Бот запущен. Пробел — пауза/продолжить, Ctrl+C — полная остановка.")
    listener = start_pause_listener()
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nБот остановлен.")
    finally:
        listener.stop()
