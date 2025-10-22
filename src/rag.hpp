#pragma once
#include "cpr/cprtypes.h"
#include "vector_db.hpp"
#include <string>
#include <vector>

const static cpr::Url model_address{"http://localhost:10101/completion"};
const static cpr::Url embeder_address("http://localhost:10100/embedding");

class Rag
{

    /**
   * @brief Список хранящий все векторные БД.
   */
    std::vector<VectorDatabase> vector_database_list;
    int dim = 0;

public:
    /**
   * @brief Проверка доступа к сервера модели и эмбедера. Инициализация баз
   * @param dim - размерность векторов в БД.
   * данных.
   *
   */
    Rag(/*int dim*/);

    /**
   * @brief Функция для запроса ответа у модели.
   *
   * @param qeustion - вопрос
   * @return std::string - ответ
   */
    std::string request(std::string qeustion, std::vector<int> database_id_list);

    /**
   * @brief Добавляет документ в БД.
   *
   * @param filename - название файла
   * @param bath_size - размер батча
   * @param database_id - id БД
   */
    void addDocument(std::string filename, int batch_size, int database_id);

    /**
   * @brief Создание новой БД.
   *
   * @param filename - название файла БД
   * @param dim - размерность БД
   * @param files - файлы которые должны быть добавлены в БД
   */
    void createDatabase(std::string filename, std::vector<std::string> files);

private:
    /**
   * @brief Заполнение vector_database_list.
   */
    void initDatabaseList();
    /**
   * @brief Получить вектор для куска текста.
   *
   * @param text - текст
   * @return const std::vector<float> - вектор
   */
    const std::vector<float> embedText(std::string text);
};
