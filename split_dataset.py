#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py — распределение размеченного датасета (YOLO) по выборкам.

Берёт пары <имя>.jpg + <имя>.txt из папки-источника (по умолчанию ``itog``)
и раскладывает их прямо в текущую папку в структуру Ultralytics/YOLOv8:

    images/
        train/   val/
    labels/
        train/   val/
    data.yaml

Картинка и её метка всегда попадают в одну и ту же выборку:
``images/train`` <-> ``labels/train``, ``images/val`` <-> ``labels/val``.

Ключевая особенность: аугментации одной и той же исходной картинки
(суффикс ``_aug_N_M`` в имени файла) НЕ разделяются между выборками —
вся группа аугментаций уходит в одну папку. Это исключает утечку данных
(data leakage), когда почти одинаковые картинки попадают и в train, и в val.

Примеры запуска (из корня проекта c:/Users/zloym/cookie):

    # посмотреть, что произойдёт, ничего не копируя
    python split_dataset.py --dry-run

    # обычный запуск: 80 / 20, создаст images/, labels/ и data.yaml
    python split_dataset.py

    # свои проценты и перенос файлов вместо копирования
    python split_dataset.py --train 70 --val 20 --test 10 --move

    # разбиение ПО КАРТИНКАМ (без группировки аугментаций)
    python split_dataset.py --no-group
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Расширения, которые считаем картинками.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Расширение файла разметки YOLO.
LABEL_EXT = ".txt"

# Регулярка для суффикса аугментации: "_aug_<idx>_<seed>" в конце имени.
AUG_RE = re.compile(r"_aug_\d+_\d+$", re.IGNORECASE)


def base_key(stem: str) -> str:
    """Вернуть базовое имя картинки без суффикса аугментации.

    ``screenshot_2026-10-26_15-27-01_aug_2_19`` -> ``screenshot_2026-10-26_15-27-01``
    """
    return AUG_RE.sub("", stem)


def collect_pairs(src: Path) -> tuple[list[str], list[str], list[str]]:
    """Собрать полные пары картинка+метка, картинки без метки и метки без картинки."""
    images = {p.stem: p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in src.iterdir() if p.is_file() and p.suffix.lower() == LABEL_EXT}

    common = sorted(set(images) & set(labels))
    image_only = sorted(set(images) - set(labels))
    label_only = sorted(set(labels) - set(images))
    return common, image_only, label_only


def group_by_base(stems: list[str], grouped: bool) -> dict[str, list[str]]:
    """Сгруппировать имена по базовой картинке (или каждая — сама по себе)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        key = base_key(stem) if grouped else stem
        groups[key].append(stem)
    return groups


def distribute(
    groups: dict[str, list[str]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Распределить ГРУППЫ по выборкам, сохраняя группы целиком.

    Жадный алгоритм: идём по перемешанным группам и кладём каждую в ту
    выборку, которая сейчас больше всего недобирает относительно своей цели.
    Это даёт проценты, близкие к заданным, и не рвёт аугментации.
    """
    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)

    # Цели по количеству КАРТИНОК.
    total = sum(len(groups[k]) for k in keys)
    targets = {name: total * ratio for name, ratio in ratios.items()}
    assigned = {name: [] for name in ratios}
    counts = {name: 0 for name in ratios}

    for key in keys:
        size = len(groups[key])
        # Кто сильнее всего отстаёт от цели (в картинках)? Туда и кладём.
        name = max(ratios.keys(), key=lambda n: targets[n] - counts[n])
        assigned[name].append(key)
        counts[name] += size

    return assigned


def _apply_move(src_file: Path, dst_file: Path, move: bool) -> None:
    """Скопировать (или перенести) файл, создав каталоги и перезаписав старый."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src_file), str(dst_file))
    else:
        shutil.copy2(src_file, dst_file)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Раскидать YOLO-разметку по train/val/test в заданных процентах.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", default="itog", help="Папка с размеченными данными.")
    parser.add_argument(
        "--dst",
        default=".",
        help="Корень, куда создать images/ и labels/ (по умолчанию — текущая папка).",
    )
    parser.add_argument("--train", type=float, default=80, help="Процент для train.")
    parser.add_argument("--val", type=float, default=20, help="Процент для val.")
    parser.add_argument(
        "--test",
        type=float,
        default=0,
        help="Процент для test (0 — папка test не создаётся, как в практической).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed перемешивания.")
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Не группировать аугментации: делить по отдельным картинкам.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Переносить файлы вместо копирования (оригиналы исчезнут из --src).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать план, ничего не трогая на диске.",
    )
    parser.add_argument(
        "--data-yaml",
        dest="data_yaml",
        action="store_true",
        default=True,
        help="Записать dataset/data.yaml (включено по умолчанию).",
    )
    parser.add_argument(
        "--no-data-yaml",
        dest="data_yaml",
        action="store_false",
        help="Не создавать data.yaml.",
    )
    parser.add_argument(
        "--classes",
        default="class.txt",
        help="Файл со списком классов (по одному в строке).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Кириллица в stdout/stderr (cmd.exe по умолчанию cp866) — принудительно UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        print(f"[ОШИБКА] Папка-источник не найдена: {src.resolve()}")
        return 1

    ratios = {"train": args.train, "val": args.val, "test": args.test}
    if any(v < 0 for v in ratios.values()):
        print("[ОШИБКА] Проценты не могут быть отрицательными.")
        return 1
    total_pct = sum(ratios.values())
    if total_pct <= 0:
        print("[ОШИБКА] Сумма процентов должна быть больше нуля.")
        return 1
    # Нормализуем на случай 100 / 100 / 100 и т.п.
    ratios = {name: value / total_pct for name, value in ratios.items()}
    ratios = {name: value for name, value in ratios.items() if value > 0}

    common, image_only, label_only = collect_pairs(src)
    if not common:
        print(f"[ОШИБКА] В папке {src.resolve()} не найдено ни одной пары image+label.")
        return 1

    grouped = not args.no_group
    groups = group_by_base(common, grouped)
    assigned = distribute(groups, ratios, args.seed)

    # ---- Сводка по количеству ----
    total_images = sum(len(groups[k]) for k in groups)
    total_groups = len(groups)

    print("=" * 64)
    print(f"Источник : {src.resolve()}")
    print(f"Назначение: {dst.resolve()}")
    print(f"Режим    : {'по группам аугментаций' if grouped else 'по отдельным картинкам'}")
    print(f"Seed     : {args.seed}   |   Действие: {'MOVE' if args.move else 'COPY'}"
          f"{'  (DRY-RUN)' if args.dry_run else ''}")
    print("=" * 64)
    print(f"Всего картинок с разметкой : {total_images}")
    print(f"Всего групп{' (уникальных картинок)' if grouped else ''}       : {total_groups}")
    if image_only:
        print(f"[!] Картинок без .txt (пропущены)  : {len(image_only)}")
    if label_only:
        print(f"[!] .txt без картинки (пропущены)  : {len(label_only)}")
    print("-" * 64)
    print(f"{'выборка':<8}{'групп':>8}{'картинок':>10}{'факт %':>10}{'цель %':>10}")
    print("-" * 64)

    summary: dict[str, dict[str, float]] = {}
    for name in ("train", "val", "test"):
        if name not in assigned:
            continue
        keys = assigned[name]
        n_img = sum(len(groups[k]) for k in keys)
        fact = (n_img / total_images * 100) if total_images else 0.0
        tgt = ratios[name] * 100
        summary[name] = {"groups": len(keys), "images": n_img, "fact": fact, "target": tgt}
        print(f"{name:<8}{len(keys):>8}{n_img:>10}{fact:>9.1f}%{tgt:>9.1f}%")
    print("-" * 64)

    if args.dry_run:
        print("DRY-RUN: файлы не изменялись. Уберите --dry-run для реального запуска.")
        return 0

    # ---- Раскладка файлов ----
    moved = 0
    for name, keys in assigned.items():
        img_dir = dst / "images" / name
        lbl_dir = dst / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for key in keys:
            for stem in groups[key]:
                # stem совпадает с именем файла без расширения — ищем реальные файлы.
                img_src = next(
                    (src / f"{stem}{ext}" for ext in IMAGE_EXTS if (src / f"{stem}{ext}").is_file()),
                    None,
                )
                lbl_src = src / f"{stem}{LABEL_EXT}"
                if img_src is None or not lbl_src.is_file():
                    continue
                _apply_move(img_src, img_dir / img_src.name, args.move)
                _apply_move(lbl_src, lbl_dir / lbl_src.name, args.move)
                moved += 1

    print(f"Готово. Обработано картинок: {moved}.")

    # ---- Итоговая раскладка ----
    print("\nКУДА ЧТО РАСКЛАДЫВАЛОСЬ:")
    for name, st in summary.items():
        print(
            f"  images/{name}  +  labels/{name}"
            f"  ->  {int(st['images'])} картинок ({st['fact']:.1f}% от всего), "
            f"цель {st['target']:.1f}%"
        )

    # ---- data.yaml (опционально) ----
    if args.data_yaml:
        classes = []
        classes_path = Path(args.classes)
        if classes_path.is_file():
            classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = [
            "path: ./",
            "train: images/train",
            "val: images/val",
        ]
        if "test" in assigned:
            lines.append("test: images/test")
        lines.append("")
        lines.append(f"nc: {len(classes)}  # количество классов")
        quoted = ", ".join(f"'{c}'" for c in classes)
        lines.append(f"names: [{quoted}]  # названия классов")
        yaml_text = "\n".join(lines) + "\n"
        (dst / "data.yaml").write_text(yaml_text, encoding="utf-8")
        print(f"\nЗаписан {dst / 'data.yaml'} (классов: {len(classes)}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
