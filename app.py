import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Загрузка обученной модели
model = YOLO("runs/detect/cookie_detector/weights/best.pt")

# Индексы классов (совпадают с data.yaml / class.txt)
CLASS_GAME = 0         # game
CLASS_UPGRADE = 1      # upgrade
CLASS_ADVERTISING = 2  # advertising

CONF_THRESHOLD = 0.7      # порог уверенности
UPGRADE_COOLDOWN = 10.0   # проход по всем улучшениям — не чаще раза в 10 секунд

# Время последнего ПОЛНОГО прохода по улучшениям
last_upgrade_pass = 0.0


def get_detections():
    """Сделать скриншот экрана и вернуть список детекций.

    Возвращает список кортежей: (класс, center_x, center_y, confidence).
    """
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np)

    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < CONF_THRESHOLD:
                continue
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detections.append((cls, center_x, center_y, confidence))
    return detections


def click_at(x, y, times=1, delay=0.1):
    """Кликнуть в точку указанное число раз без задержек между циклами."""
    for _ in range(times):
        pyautogui.click(x, y)
        if delay:
            time.sleep(delay)


def buy_all_upgrades(detections):
    """Прокликать ВСЕ найденные улучшения по порядку сверху вниз за один проход."""
    upgrades = [(cx, cy, conf) for cls, cx, cy, conf in detections if cls == CLASS_UPGRADE]
    if not upgrades:
        return False

    # Порядок как на экране: сверху вниз по Y, затем слева направо по X
    upgrades.sort(key=lambda u: (u[1], u[0]))

    for cx, cy, conf in upgrades:
        click_at(cx, cy, times=3, delay=0.05)
        print(f"Купил улучшение: ({cx:.0f}, {cy:.0f}) conf={conf:.2f}")

    return True


def main_loop():
    global last_upgrade_pass

    while True:
        detections = get_detections()

        clicked = False

        # 1. Реклама — закрываем сразу, высший приоритет
        for cls, cx, cy, conf in detections:
            if cls == CLASS_ADVERTISING:
                click_at(cx, cy, times=3, delay=0.1)
                print(f"Закрыл рекламу: ({cx:.0f}, {cy:.0f}) conf={conf:.2f}")
                clicked = True

        # 2. Улучшения — ОДИН проход по всем по порядку, не чаще раза в 10 секунд
        now = time.time()
        if now - last_upgrade_pass >= UPGRADE_COOLDOWN:
            if buy_all_upgrades(detections):
                last_upgrade_pass = now
                clicked = True

        # 3. Игровые объекты — кликаем постоянно, без перерывов
        for cls, cx, cy, conf in detections:
            if cls == CLASS_GAME:
                click_at(cx, cy, times=10, delay=0.1)
                print(f"Клик по игре: ({cx:.0f}, {cy:.0f}) conf={conf:.2f}")
                clicked = True

        # Кликер не должен простаивать: если ничего не нашли — короткая пауза
        if not clicked:
            time.sleep(0.2)


if __name__ == "__main__":
    main_loop()
