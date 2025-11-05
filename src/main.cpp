#include "rag.hpp"
#include "vector_db.hpp"
#include <cpr/cpr.h>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#define VERSION "1.0\n"
const std::string DB_STATE[] = {"ON", "OFF"};

void CLI()
{
    std::cout << VERSION;
    try
    {
        auto rag = Rag();

        rag.createDatabase("pushkin", {"/home/kisma/Projects/rag-llm/build/test_folder/pushkin_l.txt"});
        std::unordered_set<int> enabledDB{};

        while (true)
        {
            int choice;
            std::cout << "\n1. Список баз данных\n2. Включить/Выключить базу данных\n3. Запрос\n";
            std::cin >> choice;
            switch (choice)
            {
            case 1:
            {
                int itter = 1;
                std::string state;
                for (auto vector_db : rag.get_vector_database_list())
                {

                    if (enabledDB.find(itter) != enabledDB.end())
                    {
                        state = DB_STATE[0];
                    }
                    else
                    {
                        state = DB_STATE[1];
                    }
                    std::cout << itter << ". " << vector_db.getFilename() << " (" << state << ")\n";
                    ++itter;
                }
                break;
            }
            case 2:
            {
                int id;
                std::cout << "Введите номер БД: ";
                std::cin >> id;
                if (auto pos = enabledDB.find(id); pos != enabledDB.end())
                {
                    enabledDB.emplace(id);
                }
                else
                {
                    enabledDB.erase(pos);
                }
                break;
            }
            case 3:
            {
                std::cout << "\n<q> для выхода\n";
                while (true)
                {
                    std::string req;
                    std::cout << "\nЗапрос: ";
                    std::cin >> req;
                    if (req == "q")
                    {
                        break;
                    }
                    std::cout << rag.request(req, std::vector<int>{enabledDB.begin(),enabledDB.end()});
                }
            }
            default:
            {
                std::cout << "Неверное значение!\n";
                break;
            }
            }
        }

        while (true)
        {
            std::string message;
            std::cout << "Введите запрос: " << '\n';
            std::getline(std::cin, message);
            std::cout << rag.request(message, {1}) + "\n";
        }
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}

int main()
{
}
