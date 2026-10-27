import pyautogui
import time
from datetime import datetime
import os

# Папка для сохранения скриншотов (на диске C)
save_dir = r"C:\cookie_screen"
os.makedirs(save_dir, exist_ok=True)

print("Начинаю делать скриншоты каждые 5 секунд...")
print("Нажмите Ctrl+C для остановки\n")

count = 0

try:
    while True:
        # Делаем скриншот
        screenshot = pyautogui.screenshot()

        # Формируем имя файла с датой и временем
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_dir}/screenshot_{timestamp}.png"

        # Сохраняем
        screenshot.save(filename)
        count += 1
        print(f"[{count}] Сохранено: {filename}")

        # Ждём 5 секунд
        time.sleep(5)

except KeyboardInterrupt:
    print(f"\nОстановлено. Всего сделано скриншотов: {count}")
