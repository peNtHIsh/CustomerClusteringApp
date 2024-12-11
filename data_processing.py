# data_processing.py
import pandas as pd
import numpy as np
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
import csv
import chardet
from scipy import stats


class DataProcessor:
    def load_data(self, file_path):
        try:
            if file_path.endswith('.csv'):
                # Определение кодировки файла с использованием chardet
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read(100000))
                    encoding = result['encoding']

                # Определение разделителя автоматически
                with open(file_path, 'r', encoding=encoding) as csvfile:
                    sample = csvfile.read(1024)
                    sniffer = csv.Sniffer()
                    try:
                        dialect = sniffer.sniff(sample)
                        delimiter = dialect.delimiter
                    except csv.Error:
                        delimiter = ','  # По умолчанию запятая

                # Чтение CSV с определенным разделителем
                data = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)

            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла. Пожалуйста, загрузите файл CSV или Excel.")

            if data.shape[0] < 10:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для анализа.")
                return None
            return data
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке файла: {e}")
            return None

    def preprocess_data(self, data):
        numeric_columns = self.find_numeric_columns(data)
        if numeric_columns is None:
            return None, None
        data_cleaned = data[numeric_columns]

        print("Начальные данные после выбора числовых столбцов:")
        print(data_cleaned.head())

        # Обработка пропущенных значений
        if data_cleaned.isnull().sum().sum() > 0:
            response = messagebox.askyesno(
                "Обработка пропущенных значений",
                "В данных имеются пропущенные значения. Заполнить их медианными значениями?"
            )
            if response:
                data_cleaned = data_cleaned.fillna(data_cleaned.median())
                messagebox.showinfo("Информация", "Пропущенные значения заполнены медианными значениями.")
                print("Пропущенные значения заполнены медианными значениями.")
            else:
                data_cleaned = data_cleaned.dropna()
                messagebox.showinfo("Информация", "Строки с пропущенными значениями удалены.")
                print("Строки с пропущенными значениями удалены.")

        # Проверка на наличие пропущенных значений после обработки
        if data_cleaned.isnull().sum().sum() > 0:
            messagebox.showerror("Ошибка", "После обработки пропущенных значений в данных всё ещё есть NaN.")
            print("После обработки пропущенных значений в данных всё ещё есть NaN.")
            return None, None

        # Обработка аномально больших значений (удаление выбросов на основе Z-скора)
        try:
            z_scores = np.abs(stats.zscore(data_cleaned))
            print("Z-scores вычислены успешно.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при вычислении Z-скоров: {e}")
            print(f"Ошибка при вычислении Z-скоров: {e}")
            return None, None

        # Замена возможных NaN в z_scores на 0, чтобы избежать проблем
        z_scores = np.nan_to_num(z_scores, nan=0.0)

        threshold = 3
        outliers = (z_scores > threshold).any(axis=1)
        num_outliers = np.sum(outliers)
        print(f"Обнаружено выбросов: {num_outliers}")

        if num_outliers > 0:
            response = messagebox.askyesno(
                "Обработка выбросов",
                f"В данных обнаружено {num_outliers} выбросов. Удалить их?"
            )
            if response:
                data_cleaned = data_cleaned[~outliers]
                messagebox.showinfo("Информация", f"Выбросы удалены. Оставлено {data_cleaned.shape[0]} записей.")
                print(f"Выбросы удалены. Оставлено {data_cleaned.shape[0]} записей.")
            else:
                # Альтернативный способ обработки выбросов: замена на пороговые значения
                data_cleaned = self.cap_outliers(data_cleaned, z_scores, threshold)
                messagebox.showinfo("Информация", "Выбросы заменены на пороговые значения.")
                print("Выбросы заменены на пороговые значения.")

        # Проверка на наличие выбросов после обработки
        try:
            z_scores_after = np.abs(stats.zscore(data_cleaned))
            z_scores_after = np.nan_to_num(z_scores_after, nan=0.0)
            if np.any(z_scores_after > threshold):
                messagebox.showwarning("Предупреждение", "В данных всё ещё присутствуют выбросы.")
                print("В данных всё ещё присутствуют выбросы.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при проверке выбросов: {e}")
            print(f"Ошибка при проверке выбросов: {e}")

        # Масштабирование данных
        scaler = StandardScaler()
        try:
            scaled_data = scaler.fit_transform(data_cleaned)
            print("Данные масштабированы успешно.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при масштабировании данных: {e}")
            print(f"Ошибка при масштабировании данных: {e}")
            return None, None

        # Проверка на наличие NaN после масштабирования
        if np.isnan(scaled_data).any():
            messagebox.showerror("Ошибка", "После масштабирования данные содержат NaN.")
            print("После масштабирования данные содержат NaN.")
            return None, None

        print("Масштабированные данные:")
        print(scaled_data[:5])

        return scaled_data, data_cleaned.columns

    def find_numeric_columns(self, data):
        # Исключение идентификаторных столбцов на основе точных названий
        exclude_columns_exact = ['CustomerID', 'InvoiceNo', 'ID', 'No', 'Date', 'Time', 'Dt_Customer']
        exclude_columns = [col for col in data.columns if col in exclude_columns_exact]

        # Оставшиеся столбцы пытаемся преобразовать в числовые
        potential_numeric = [col for col in data.columns if col not in exclude_columns]
        numeric_columns = []
        for col in potential_numeric:
            # Если столбец уже числовой
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_columns.append(col)
            else:
                # Попытка конвертации в числовой тип
                converted_col = pd.to_numeric(data[col], errors='coerce')
                if converted_col.notnull().sum() >= 2:  # Требуется минимум 2 ненулевых значения
                    data[col] = converted_col
                    numeric_columns.append(col)

        if len(numeric_columns) < 2:
            messagebox.showerror("Ошибка", "Недостаточно числовых столбцов для кластеризации.")
            print("Недостаточно числовых столбцов для кластеризации.")
            return None

        # Вывод найденных числовых столбцов для отладки
        print("Найденные числовые столбцы для кластеризации:", numeric_columns)

        return numeric_columns

    def cap_outliers(self, data, z_scores, threshold):
        """
        Замена выбросов на пороговые значения.
        """
        data_capped = data.copy()
        for col in data.columns:
            upper_limit = data[col].mean() + threshold * data[col].std()
            lower_limit = data[col].mean() - threshold * data[col].std()
            data_capped[col] = np.where(data[col] > upper_limit, upper_limit,
                                        np.where(data[col] < lower_limit, lower_limit, data[col]))
        return data_capped
