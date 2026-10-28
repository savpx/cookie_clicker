"""Cookie Clicker Bot — детекция экрана через YOLO и автоматические клики.

Логика работы цикла:
  1. Реклама      — закрывается сразу, как только появилась (высший приоритет).
                    Клик ОДИНОЧНЫЙ, с проверкой результата: если реклама не
                    исчезла — повторяем (без «пулемётной» серии кликов).
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
# Путь по умолчанию (используется, если автопоиск ничего не найдёт).
DEFAULT_MODEL_PATH = "runs/detect/cookie_detector/weights/best.pt"


def find_latest_weights():
    """Найти самый свежий best.pt среди runs/detect/*/weights/.

    YOLO при каждом обучении создаёт новую папку (cookie_detector,
    cookie_detector-2, ...), поэтому вместо жёсткого пути берём самую
    недавно обученную модель — так бот всегда использует актуальные веса.
    """
    candidates = glob.glob(os.path.join("runs", "detect", "*", "weights", "best.pt"))
    if not candidates:
        return DEFAULT_MODEL_PATH
    return max(candidates, key=os.path.getmtime)


MODEL_PATH = find_latest_weights()
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
CONF_THRESHOLDS = {
    CLASS_GAME: 0.55,
    CLASS_UPGRADE: 0.40,
    CLASS_ADVERTISING: 0.50,
}
CONF_THRESHOLD = 0.50  # запасной порог для неизвестных классов

# ---------------------------------------------------------------------------
# Настройки поведения
# ---------------------------------------------------------------------------
UPGRADE_INTERVAL = 7.0   # полный проход по улучшениям — раз в 7 секунд
UPGRADE_CLICKS = 2       # кликов по каждому улучшению (чтобы покупка точно прошла)
UPGRADE_MAX_STEPS = 20   # защита от зацикливания во время прохода
CLICKS_PER_BURST = 25    # сколько быстрых кликов по печеньке до новых скриншотов

# --- Реклама ---
AD_MAX_ATTEMPTS = 4      # максимум попыток закрыть одну рекламу
AD_ATTEMPT_DELAY = 0.35  # пауза между одиночными кликами (чтобы интерфейс успел)
AD_SETTLE = 0.30         # пауза после закрытия рекламы перед другими действиями
# Куда кликать внутри рамки рекламы: (0.5, 0.5) — центр.
# Если ваша разметка — это весь баннер, а крестик в правом верхнем углу,
# попробуйте (0.92, 0.08).
AD_CLICK_POS = (0.5, 0.5)

# Диагностика: раз в секунду печатать, сколько объектов каждого класса найдено.
DEBUG = True

# Детекция: класс + центр + уверенность + рамка
Detection = namedtuple("Detection", "cls cx cy conf x1 y1 x2 y2")

# Время последнего полного прохода по улучшениям
last_upgrade_pass = 0.0

# Флаг паузы: переключается нажатием пробела
paused = False

# Для ограничения частоты отладочных сообщений
_last_debug_log = 0.0


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
            detections.append(
                Detection(
                    cls=cls,
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                    conf=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
    return detections


def log_debug(detections):
    """Не чаще раза в секунду показать, сколько объектов найдено по классам."""
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


def close_ads(detections):
    """Закрыть рекламу: один клик -> проверка -> при необходимости повтор.

    Раньше делалось 3 мгновенных клика подряд: если реклама закрывалась с
    первого, остальные попадали по игре/ссылке под ней. Теперь кликаем по
    одному разу и после каждого клика проверяем, исчезла ли реклама.

    Возвращает True, если реклама была и её удалось закрыть.
    """
    ads = [d for d in detections if d.cls == CLASS_ADVERTISING]
    if not ads:
        return False

    # Самое уверенное окно — первым.
    ads.sort(key=lambda d: d.conf, reverse=True)
    ad = ads[0]

    # Точка клика внутри рамки (по умолчанию — центр).
    px = ad.x1 + (ad.x2 - ad.x1) * AD_CLICK_POS[0]
    py = ad.y1 + (ad.y2 - ad.y1) * AD_CLICK_POS[1]

    for attempt in range(1, AD_MAX_ATTEMPTS + 1):
        wait_if_paused()
        pyautogui.click(px, py)
        time.sleep(AD_ATTEMPT_DELAY)

        still_there = any(d.cls == CLASS_ADVERTISING for d in get_detections())
        if not still_there:
            print(f"Закрыл рекламу за {attempt} клик(ов): ({px:.0f}, {py:.0f}) conf={ad.conf:.2f}")
            return True

    print(f"Реклама не поддалась за {AD_MAX_ATTEMPTS} попыток — пропускаю до следующего кадра.")
    return False


def buy_all_upgrades():
    """Проход по улучшениям: СНИЗУ ВВЕРХ, затем СЛЕВА НАПРАВО.

    Детекции обновляются ПЕРЕД КАЖДЫМ кликом — координаты всегда актуальны,
    поэтому улучшение (например, «Курсор») не пропускается из-за устаревшего
    кадра. Возвращает True, если что-то было куплено.
    """
    bought = False
    clicked = set()

    for _ in range(UPGRADE_MAX_STEPS):
        wait_if_paused()
        upgrades = [d for d in get_detections() if d.cls == CLASS_UPGRADE]
        if not upgrades:
            break

        # y-координаты растут сверху вниз: -cy даёт обход снизу вверх,
        # внутри одного ряда — слева направо по cx.
        upgrades.sort(key=lambda d: (-d.cy, d.cx))

        # Берём ближайшее улучшение, которое ещё не кликали в этом проходе.
        target = None
        for det in upgrades:
            key = (round(det.cx / 5), round(det.cy / 5))
            if key not in clicked:
                target = det
                clicked.add(key)
                break

        if target is None:
            # Все найденные улучшения уже обработаны (например, не хватает денег).
            break

        click_at(target.cx, target.cy, times=UPGRADE_CLICKS, delay=0.05)
        print(f"Купил улучшение: ({target.cx:.0f}, {target.cy:.0f}) conf={target.conf:.2f}")
        bought = True

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
