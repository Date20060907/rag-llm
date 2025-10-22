#include "vector_db.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>


namespace fs = std::filesystem;

VectorDatabase::VectorDatabase(const std::string &db_filename, size_t dim)
    : filename(db_filename), dimension(dim), modified(false)
{
}

VectorDatabase::~VectorDatabase()
{
    if (modified)
    {
        save();
    }
}

bool VectorDatabase::initialize()
{
    fs::path new_folder = "db";

    try
    {
        fs::create_directory(new_folder);
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Ошибка: " << e.what() << '\n';
    }

    if (load())
    {
        std::cout << "База данных загружена: " << vectors.size() << " векторов" << std::endl;
        return true;
    }

    std::ofstream file("./db/" + filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Ошибка создания файла базы данных" << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char *>(&dimension), sizeof(size_t));
    uint32_t num_vectors = 0;
    file.write(reinterpret_cast<const char *>(&num_vectors), sizeof(uint32_t));

    std::cout << "Создана новая база данных, размерность: " << dimension << std::endl;
    return true;
}

uint32_t VectorDatabase::addEmbedding(const std::vector<float> &embedding, const std::string &metadata)
{
    if (embedding.size() != dimension)
    {
        std::cerr << "Ошибка: размерность эмбеддинга (" << embedding.size() << ") не совпадает с размерностью БД ("
                  << dimension << ")" << std::endl;
        return 0;
    }

    VectorRecord record;
    record.id = generateId();
    record.embedding = embedding;
    record.metadata = metadata;

    normalizeVector(record.embedding);

    vectors.push_back(record);
    id_to_index[record.id] = vectors.size() - 1;
    modified = true;

    std::cout << "Добавлен вектор ID: " << record.id << std::endl;
    return record.id;
}

std::vector<std::pair<uint32_t, float>> VectorDatabase::findTopK(const std::vector<float> &query,
                                                                 uint32_t k,
                                                                 float similarity_threshold)
{

    if (query.size() != dimension)
    {
        std::cerr << "Ошибка: размерность запроса не совпадает с размерностью БД" << std::endl;
        return {};
    }

    if (vectors.empty())
    {
        return {};
    }

    std::vector<float> normalized_query = query;
    normalizeVector(normalized_query);

    std::vector<std::pair<uint32_t, float>> similarities;

    for (const auto &record : vectors)
    {
        float similarity = cosineSimilarity(normalized_query, record.embedding);

        if (similarity >= similarity_threshold)
        {
            similarities.push_back(std::make_pair(record.id, similarity));
        }
    }

    std::sort(
        similarities.begin(),
        similarities.end(),
        [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) { return a.second > b.second; });

    if (k > similarities.size())
    {
        k = static_cast<uint32_t>(similarities.size());
    }

    return std::vector<std::pair<uint32_t, float>>(similarities.begin(), similarities.begin() + k);
}

float VectorDatabase::cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) const
{
    float dot_product = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot_product += a[i] * b[i];
    }
    return dot_product;
}

void VectorDatabase::normalizeVector(std::vector<float> &vector) const
{
    float norm = 0.0f;
    for (float value : vector)
    {
        norm += value * value;
    }

    if (norm > 0.0f)
    {
        norm = std::sqrt(norm);
        for (float &value : vector)
        {
            value /= norm;
        }
    }
}

uint32_t VectorDatabase::generateId() const
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint32_t> dis(1000, 9999999);

    return dis(gen);
}

bool VectorDatabase::save()
{
    std::ofstream file("./db/" + filename, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        std::cerr << "Ошибка сохранения базы данных" << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char *>(&dimension), sizeof(size_t));
    uint32_t num_vectors = static_cast<uint32_t>(vectors.size());
    file.write(reinterpret_cast<const char *>(&num_vectors), sizeof(uint32_t));

    for (const auto &record : vectors)
    {
        file.write(reinterpret_cast<const char *>(&record.id), sizeof(uint32_t));

        uint32_t metadata_size = static_cast<uint32_t>(record.metadata.size());
        file.write(reinterpret_cast<const char *>(&metadata_size), sizeof(uint32_t));
        if (metadata_size > 0)
        {
            file.write(record.metadata.c_str(), metadata_size);
        }

        file.write(reinterpret_cast<const char *>(record.embedding.data()), sizeof(float) * dimension);
    }

    modified = false;
    std::cout << "База данных сохранена: " << vectors.size() << " векторов" << std::endl;
    return true;
}

bool VectorDatabase::load()
{
    std::ifstream file("./db/" + filename, std::ios::binary);
    if (!file)
    {
        return false;
    }

    size_t file_dimension;
    file.read(reinterpret_cast<char *>(&file_dimension), sizeof(size_t));

    if (file_dimension != dimension)
    {
        std::cerr << "Ошибка: размерность в файле не совпадает с ожидаемой" << std::endl;
        return false;
    }

    uint32_t num_vectors;
    file.read(reinterpret_cast<char *>(&num_vectors), sizeof(uint32_t));

    vectors.clear();
    vectors.reserve(num_vectors);
    id_to_index.clear();

    for (uint32_t i = 0; i < num_vectors; ++i)
    {
        VectorRecord record;

        file.read(reinterpret_cast<char *>(&record.id), sizeof(uint32_t));

        // ����������
        uint32_t metadata_size;
        file.read(reinterpret_cast<char *>(&metadata_size), sizeof(uint32_t));
        if (metadata_size > 0)
        {
            record.metadata.resize(metadata_size);
            file.read(&record.metadata[0], metadata_size);
        }

        // ������
        record.embedding.resize(dimension);
        file.read(reinterpret_cast<char *>(record.embedding.data()), sizeof(float) * dimension);

        vectors.push_back(record);
        id_to_index[record.id] = vectors.size() - 1;
    }

    modified = false;
    return true;
}

std::string VectorDatabase::getMetadata(uint32_t id) const
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end() && it->second < vectors.size())
    {
        return vectors[it->second].metadata;
    }
    return "";
}

bool VectorDatabase::updateMetadata(uint32_t id, const std::string &new_metadata)
{
    auto it = id_to_index.find(id);
    if (it != id_to_index.end() && it->second < vectors.size())
    {
        vectors[it->second].metadata = new_metadata;
        modified = true;
        return true;
    }
    return false;
}
