#include "rag.hpp"
#include "cpr/api.h"
#include "json.hpp"
#include "vector_db.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define BATCH 1024
#define DEBUG


Rag::Rag(/*int dim*/)
{
    if (auto r = cpr::Head(model_address); r.status_code == 0)
    {
        std::cerr << "Failed to init connection to model!\n";
        throw std::runtime_error(r.error.message);
    }
    if (auto r = cpr::Head(embeder_address); r.status_code == 0)
    {
        std::cerr << "Failed to init connection to embedder!\n";
        throw std::runtime_error(r.error.message);
    }

    this->dim = dim;
    this->dim = static_cast<int>(this->embedText("Test text").size());
    std::cerr << "dim is setted!\n";
    initDatabaseList();
}

std::string Rag::request(std::string question,
                         std::vector<int> database_id_list,
                         int n_predict,
                         float temperature,
                         int top_k,
                         int rag_k,
                         float rag_sim_threshold)
{
    std::string context = "Контекст из базы данных для использования в ответе:\n";

    auto embeded_question = this->embedText(question);
    std::cout << database_id_list.size() << std::endl;
    for (auto db_id : database_id_list)
    {
        std::cout << "\nSelected ID" << db_id << std::endl;
        auto temp = this->vector_database_list[db_id].findTopK(embeded_question, rag_k, rag_sim_threshold);
        std::cout << temp.size() << std::endl;
        for (auto id : temp)
        {
#ifdef DEBUG
            std::cout << id.second << std::endl;
            std::cout << vector_database_list[db_id].getMetadata(id.first) << std::endl;
#endif // DEBUG
            context += vector_database_list[db_id].getMetadata(id.first) + "\n";
        }
    }

#ifdef DEBUG
    std::cout << context + "Запрос пользователя:\n" + question << std::endl;
#endif // DEBUG


    nlohmann::json payload = {{"prompt",
                               "<|im_start|>system\n Ты - полезный AI-ассистент. Ответь на вопрос пользователя, "
                               "используя ТОЛЬКО предоставленную информацию из базы знаний. Если в предоставленной "
                               "информации нет достаточных данных для ответа, честно скажи об этом. Будь точным, "
                               "кратким и используй только факты из контекста.\n<|im_end|>\n<|im_start|>user\n" +
                                   context + "Вопрос: " + question +
                                   "Основываясь на предоставленной информации выше, дай точный и краткий "
                                   "ответ.\n<|im_end|>\n<|im_start|>assistant\n"
                            // "<|im_start|>system\n Ты - персонаж, характеристика которого будет передана тебе в виде контекста. Твоя задача мимкрировать под персонажа и отвечать пользователю от его лица. Отвечай разговорной речью как будто в переписке в социальной сети\n<|im_end|>\n<|im_start|>user\n" +
                            //        context + "Вопрос: " + question +
                            //        "<|im_end|>\n<|im_start|>assistant\n"
                            },

                              {"n_predict", n_predict},
                              {"temperature", temperature},
                              {"top_k", top_k},
                              {"stream", false},
                              {"stop", {"<|im_end|>"}}};

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

void Rag::addDocument(std::string filename, int batch_size, int database_id)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::vector<std::string> chunks;
    size_t pos = 0;

    while (pos < content.size())
    {
        size_t start = pos;
        int char_count = 0;

        while (pos < content.size() && char_count < batch_size)
        {
            auto c = static_cast<unsigned char>(content[pos]);
            if ((c & 0x80) == 0)
            {
                pos += 1;
            }
            else if ((c & 0xE0) == 0xC0)
            {
                pos += 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                pos += 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                pos += 4;
            }
            else
            {
                throw std::runtime_error("Invalid UTF-8 sequence");
            }
            ++char_count;
        }

        chunks.push_back(content.substr(start, pos - start));
    }

    for (const auto &chunk : chunks)
    {
        auto embeding = embedText(chunk);
        vector_database_list[database_id].addEmbedding(embeding, chunk);
    }
}

void Rag::addDocumentByParagraphs(std::string filename, int database_id)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    std::string paragraph;

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            if (!paragraph.empty())
            {
                auto embedding = embedText(paragraph);
#ifdef DEBUG
                std::cout << paragraph << std::endl;
#endif // DEBUG
                vector_database_list[database_id].addEmbedding(embedding, paragraph);
                paragraph.clear();
            }
        }
        else
        {
            if (!paragraph.empty())
            {
                paragraph += "\n";
            }
            paragraph += line;
        }
    }
    if (!paragraph.empty())
    {
        auto embedding = embedText(paragraph);
        vector_database_list[database_id].addEmbedding(embedding, paragraph);
    }

    file.close();
}

void Rag::createDatabase(std::string filename, std::vector<std::string> files, generatorType type)
{
    VectorDatabase new_db{filename, static_cast<size_t>(dim)};
    if (!new_db.initialize())
    {
        throw std::runtime_error("Failed to initialize database");
    }
    this->vector_database_list.push_back(new_db);

#ifdef DEBUG
    std::cout << "\ntype: " << type << std::endl;
#endif // DEBUG

    for (auto file : files)
    {
        switch (type)
        {
        case generatorType::chunk:
            this->addDocument(file, BATCH, static_cast<int>(this->vector_database_list.size() - 1));
            break;
        case generatorType::paragraphs:
            this->addDocumentByParagraphs(file, static_cast<int>(this->vector_database_list.size() - 1));
            break;
        default:
            break;
        }
    }
}

void Rag::initDatabaseList()
{
    const std::string db_path = "./db";
    if (!std::filesystem::exists(db_path) || !std::filesystem::is_directory(db_path))
    {
        std::cerr << "Directory ./db does not exist or is not a directory." << std::endl;
        return;
    }
    for (const auto &entry : std::filesystem::directory_iterator(db_path))
    {
        if (entry.is_regular_file())
        {
            std::cout << entry.path() << std::endl;
            vector_database_list.emplace_back(entry.path(), dim);
            if (!vector_database_list.back().initialize())
            {
                vector_database_list.pop_back();
            }
        }
    }
}

const std::vector<float> Rag::embedText(std::string text)
{
    std::cout << "Start embedding\n";
    nlohmann::json payload;
    payload["content"] = text;
    cpr::Response r = cpr::Post(cpr::Url(embeder_address),
                                cpr::Header{{"Content-Type", "application/json"}},
                                cpr::Body{payload.dump()});
    if (r.status_code != 200)
    {
        throw std::runtime_error(r.error.message);
    }

    try
    {
        auto response_json = nlohmann::json::parse(r.text);
        std::vector<float> m;

        if (response_json.is_array() && !response_json.empty() && response_json[0].contains("embedding") &&
            response_json[0]["embedding"].is_array() && !response_json[0]["embedding"].empty() &&
            response_json[0]["embedding"][0].is_array())
        {
            for (const auto &val : response_json[0]["embedding"][0])
            {
                if (val.is_number())
                {
                    m.push_back(val.get<float>());
                }
                else
                {
                    throw std::runtime_error("Embedding value is not a number");
                }
            }
        }
        else
        {
            std::cerr << response_json << std::endl;
            throw std::runtime_error("Invalid response format: expected array of embeddings");
        }

        if (m.size() != this->dim && dim > 0)
        {
            throw std::runtime_error("Embedder returned vector with wrong size: " + std::to_string(m.size()) +
                                     ". Expected: " + std::to_string(this->dim));
        }

        std::cout << "finished embedding!\n" << m.data() << "\n" << std::endl;
        return m;
    }
    catch (const nlohmann::json::exception &e)
    {
        throw std::runtime_error(std::string("JSON parse error: ") + e.what());
    }
}

const std::vector<VectorDatabase> &Rag::get_vector_database_list() const
{
    return vector_database_list;
}
