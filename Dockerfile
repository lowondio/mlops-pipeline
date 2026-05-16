# 1. Берем официальный легкий образ Python
FROM python:3.10-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файл с зависимостями в контейнер
COPY requirements.txt .

# 4. Устанавливаем библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем весь наш код и данные в контейнер
COPY src/ ./src/
COPY data/ ./data/

# 6. Команда по умолчанию: запускаем обучение
CMD ["python", "src/train_model.py"]