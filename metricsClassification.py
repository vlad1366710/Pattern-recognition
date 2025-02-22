
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Загрузка данных
data = pd.read_csv('iris.csv.gz')

# Разделение данных на признаки (X) и метки классов (y)
X = data.iloc[:, :-1].values  # Извлекаем все столбцы, кроме последнего, как признаки
y = data.iloc[:, -1].values  # Извлекаем последний столбец как метки классов

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=23)


# Реализация метрик
def euclidean_distance(x1, x2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def hamming_distance(x1, x2):
    """Вычисляет расстояние Хэмминга между двумя точками.
    Расстояние Хэмминга определяет количество позиций, в которых соответствующие элементы двух векторов различаются."""
    return np.sum(x1 != x2)

def manhattan_distance(x1, x2):
    """Вычисляет манхэттенское расстояние между двумя точками.
    многомерном пространстве, основанная на сумме абсолютных разностей их координат."""
    return np.sum(np.abs(x1 - x2))

def jaccard_similarity(x1, x2):
    """Вычисляет коэффициент Жаккара между двумя точками.
    коэффициент — это мера сходства между двумя наборами данных, которая определяется как отношение размера их пересечения к размеру их объединения."""
    intersection = np.sum(np.minimum(x1, x2))  # Находим пересечение
    union = np.sum(np.maximum(x1, x2))  # Находим объединение
    return 1 - (intersection / union)  # Возвращаем отношение

def cosine_similarity(x1, x2):
    """Вычисляет косинусное сходство между двумя точками.
    Косинусное сходство — это мера сходства между двумя ненулевыми векторами, определяемая косинусом угла между ними."""
    dot_product = np.dot(x1, x2)  # Скалярное произведение
    norm_x1 = np.linalg.norm(x1)  # Норму первого вектора
    norm_x2 = np.linalg.norm(x2)  # Норму второго вектора
    return 1 - (dot_product / (norm_x1 * norm_x2))  # В данной функции возвращается значение 1−J, что означает, что функция возвращает "дистанцию" между двумя наборами, где 0 указывает на полное совпадение, а 1 — на полное несоответствие.

# Функция классификации на основе ближайшего соседа
def classify(X_train, y_train, x_test, metric):
    """Классифицирует тестовую точку на основе ближайшего соседа."""
    distances = []  # Список для хранения расстояний до обучающих точек
    for i in range(len(X_train)):
        dist = metric(X_train[i], x_test)  # Вычисляем расстояние до каждой обучающей точки
        distances.append((dist, y_train[i]))  # Сохраняем расстояние и соответствующую метку класса
    distances.sort(key=lambda x: x[0])  # Сортируем по расстоянию
    return distances[0][1]  # Возвращаем метку класса ближайшего соседа

# Тестирование на тестовой выборке
def test_classification(X_train, y_train, X_test, y_test, metric):
    """
    Функция для тестирования классификации на тестовой выборке.

    Параметры:
    X_train (array-like): Обучающая выборка (признаки).
    y_train (array-like): Метки классов для обучающей выборки.
    X_test (array-like): Тестовая выборка (признаки).
    y_test (array-like): Метки классов для тестовой выборки.
    metric (function): Функция для вычисления расстояния или схожести.

    Возвращает:
    float: Доля правильно классифицированных примеров (точность).
    """
    correct = 0  # Счетчик правильных предсказаний
    for i in range(len(X_test)):
        # Предсказание класса для текущего примера тестовой выборки
        predicted = classify(X_train, y_train, X_test[i], metric)
        # Проверка, совпадает ли предсказанный класс с истинным
        if predicted == y_test[i]:
            correct += 1  # Увеличиваем счетчик, если предсказание верное
    accuracy = correct / len(X_test)  # Вычисляем точность
    return accuracy  # Возвращаем точность


metrics = {
    'Euclidean': euclidean_distance,
    'Hamming': hamming_distance,
    'Manhattan': manhattan_distance,
    'Jaccard': jaccard_similarity,
    'Cosine': cosine_similarity
}

# Цикл по метрикам для оценки точности классификации
for metric_name, metric_func in metrics.items():
    accuracy = test_classification(X_train, y_train, X_test, y_test, metric_func)
    print(f'{metric_name} Accuracy: {accuracy:.2f}')  # Выводим результаты точности