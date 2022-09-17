# BoW-MLP


# Запуск

Файл для запуска: executor/executor.py

Запустится валидация на train, test данных с лучшим чекпоинтом и затем продолжится обучение.

Код для запуска на кэггл: https://www.kaggle.com/rvnrvn1/bow-mlp

Графики: saved_files/plots/accuracy_loss (all experiments)/

Конф. матрицы: saved_files/plots/conf_matrices/

Mlflow логи: executor/mlruns.zip

Лучшая точность: 90.5 %

Testing error: 9.5 %

Лучшая точность получена с 2 fc слоями: 50000x128, 128x4 с дропаутами, функцией активации ReLU, xavier инициализацией весов (название эксперимента в коде и папках с визуализацией: 1_hidden_layer_128_dim).
