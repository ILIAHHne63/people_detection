import argparse
import cv2
from ultralytics import YOLO


def process_video(input_path: str, output_path: str, model_weights: str = "yolo12x.pt") -> None:
    """
    Обрабатывает видео, выполняя детекцию людей с использованием модели YOLO.

    Args:
        input_path (str): Путь к исходному видеофайлу.
        output_path (str): Путь для сохранения обработанного видео.
        model_weights (str): Путь к файлу весов модели YOLO. По умолчанию "yolo12x.pt".
    """
    # Загрузка модели YOLO
    model = YOLO(model_weights)

    # Открытие видеофайла
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Получение параметров видео (FPS, ширина, высота)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Обработка видео по кадрам
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Детекция и трекинг объектов (только класс 0 - люди)
            results = model.track(frame, persist=True, verbose=False, classes=[0])

            # Отрисовка результатов на кадре
            annotated_frame = results[0].plot()

            # Запись кадра в выходной файл
            out.write(annotated_frame)

            # Выход по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Обработанное видео сохранено в: {output_path}")


def main() -> None:
    """
    Основная функция программы. Запускает обработку видео.
    """
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Детекция людей на видео с использованием модели YOLO.")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Путь к исходному видеофайлу."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Путь для сохранения обработанного видео."
    )
    parser.add_argument(
        "-m", "--model",
        default="yolo12x.pt",
        help="Путь к файлу весов модели YOLO. По умолчанию 'yolo12x.pt'."
    )
    args = parser.parse_args()

    # Запуск обработки видео
    process_video(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
