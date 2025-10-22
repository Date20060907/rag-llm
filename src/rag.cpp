#include "rag.hpp"
#include "cpr/api.h"
#include "json.hpp"
#include "vector_db.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#define BATCH 1024
#define DEBUG


Rag::Rag(/*int dim*/)
{
    if (auto r = cpr::Head(model_address); r.status_code == 0)
    {
        throw std::runtime_error(r.error.message);
    }
    if (auto r = cpr::Head(embeder_address); r.status_code == 0)
    {
        throw std::runtime_error(r.error.message);
    }

    // this->dim = dim;
    this->dim = static_cast<int>(this->embedText("Test text").size());
    initDatabaseList();
}

std::string Rag::request(std::string qeustion, std::vector<int> database_id_list)
{
    std::string context = "Контекст:\n";

    auto embeded_question = this->embedText(qeustion);
    for (auto db_id : database_id_list)
    {
        auto temp = this->vector_database_list[db_id].findTopK(embeded_question, 5, 0.2);
        for (auto id : temp)
        {
#ifdef DEBUG
            std::cout << id.second << std::endl;
#endif // DEBUG
            context += vector_database_list[db_id].getMetadata(id.first) + "\n";
        }
    }

#ifdef DEBUG
    std::cout << context + "Вопрос:\n" + qeustion << std::endl;
#endif // DEBUG


    nlohmann::json payload = {{"prompt", context + "Вопрос:\n" + qeustion},
                              {"n_predict", 500},
                              {"temperature", 0.1},
                              {"top_k", 5},
                              {"stream", false}};

    cpr::Response r =
        cpr::Post(model_address, cpr::Header{{"Content-Type", "application/json"}}, cpr::Body{payload.dump()});

    if (r.status_code == 200)
    {
        try
        {
            auto response_json = nlohmann::json::parse(r.text);
            return response_json.value("content", "");
        }
        catch (const std::exception &e)
        {
            return "";
        }
    }
    return "";
}

//Навайбкодило
void Rag::addDocument(std::string filename, int batch_size, int database_id)
{
    std::ifstream input_file(filename, std::ios::binary);

#ifdef DEBUG
    std::cout << input_file.is_open() << std::endl;
#endif // DEBUG
    if (!input_file)
    {
#ifdef DEBUG
        std::cout << "ERROR WHILE READING A FILE!" << '\n';
#endif // DEBUG
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Получаем базовое имя файла без пути (опционально, можно оставить полный путь)
    std::string base_name = filename;
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos)
    {
        base_name = filename.substr(last_slash + 1);
    }

    std::vector<char> buffer(batch_size);
    int batch_number = 0;

    while (input_file)
    {
        input_file.read(buffer.data(), batch_size);
        std::streamsize bytes_read = input_file.gcount();

        if (bytes_read == 0)
        {
            break;
        }
        this->vector_database_list[database_id].addEmbedding(this->embedText(buffer.data()), buffer.data());

        ++batch_number;
    }

    input_file.close();
}

void Rag::createDatabase(std::string filename, std::vector<std::string> files)
{
    VectorDatabase new_db{filename, static_cast<size_t>(dim)};
    if (!new_db.initialize())
    {
        throw std::runtime_error("Failed to initialize database");
    }
    this->vector_database_list.push_back(new_db);

    for (auto file : files)
    {
        this->addDocument(file, BATCH, static_cast<int>(this->vector_database_list.size() - 1));
    }
}

void Rag::initDatabaseList()
{
    for (const auto &entry : std::filesystem::directory_iterator())
    {
        if (entry.is_regular_file())
        {
            vector_database_list.emplace_back(entry.path(), dim);
            if (!vector_database_list.end()->initialize())
            {
                vector_database_list.pop_back();
            }
        }
    }
}

const std::vector<float> Rag::embedText(std::string text)
{
    nlohmann::json payload;
    payload["content"] = text; // ← всё экранируется автоматически!

    cpr::Response r = cpr::Post(cpr::Url("http://localhost:10100/embedding"),
                                cpr::Header{{"Content-Type", "application/json"}},
                                cpr::Body{payload.dump()});
    if (r.status_code != 200)
    {
        throw std::runtime_error(r.error.message);
    }
    std::string embeder = r.text;
    std::vector<float> m;
    int parthent = 0;
    parthent = embeder.find('[');
    parthent = embeder.find('[', parthent + 1);
    parthent = embeder.find('[', parthent + 1);
    int pos1 = parthent;
    int pos2 = embeder.find(',', pos1 + 1);
    while (pos2 != std::string::npos)
    {
        m.emplace_back(std::stof(embeder.substr(pos1 + 1, pos2 - pos1 - 1)));
        pos1 = pos2;
        pos2 = embeder.find(',', pos2 + 1);
    }

    m.emplace_back(std::stof(embeder.substr(pos1 + 1, embeder.find(']', pos1 + 1) - pos1 - 1)));
    if (m.size() != this->dim && dim > 0)
    {
        throw std::runtime_error("Embeder returned vector with wrong size: " + std::to_string(m.size()) +
                                 ". Expected: " + std::to_string(this->dim));
    }
    return m;
}
