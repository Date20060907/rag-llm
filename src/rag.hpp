#pragma once
#include "cpr/cprtypes.h"
#include "vector_db.hpp"
#include <string>
#include <vector>

const static cpr::Url model_address{"100.124.183.1:10101/completion"};
const static cpr::Url embeder_address("100.124.183.1:10100/embedding");

enum generatorType{
    chunk,
    paragraphs
};

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
   * @param question - вопрос
   * @param database_id_list - список идентификаторов баз данных для поиска контекста
   * @param n_predict - максимальное количество токенов в ответе
   * @param temperature - температура выборки (влияет на креативность ответа)
   * @param top_k - количество наиболее вероятных токенов для ограничения выборки
   * @return std::string - ответ
   */
    std::string request(std::string question,
                        std::vector<int> database_id_list,
                        int n_predict = 500,
                        float temperature = 0.5,
                        int top_k = 1,
                        int rag_k = 3,
                        float rag_sim_threshold = 0.3f);

    /**
   * @brief Добавляет документ в БД.
   *
   * @param filename - название файла
   * @param bath_size - размер батча
   * @param database_id - id БД
   */
    void addDocument(std::string filename, int batch_size, int database_id);

    /**
     * @brief Добавляет документ в БД, разбивая его на параграфы по пустым строкам.
     *
     * @param filename - название файла
     * @param database_id - id БД
     */

    void addDocumentByParagraphs(std::string filename, int database_id);

    /**
   * @brief Создание новой БД.
   *
   * @param filename - название файла БД
   * @param dim - размерность БД
   * @param files - файлы которые должны быть добавлены в БД
   */
    void createDatabase(std::string filename, std::vector<std::string> files, generatorType type);

    /**
     * @brief Получить массив std::vector<VectorDatabase>
     *
     * @return std::vector<VectorDatabase>
     */
    const std::vector<VectorDatabase> &get_vector_database_list() const;

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
