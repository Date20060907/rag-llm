#include "rag.hpp"
#include "vector_db.hpp"
#include <cpr/cpr.h>
#include <exception>
#include <iostream>
#include <string>

#define VERSION "1.0\n"

int main()
{
    std::cout << VERSION;
    try
    {
        auto rag = Rag();
        rag.createDatabase("pushkin", {"/home/kisma/Projects/rag-llm/build/test_folder/pushkin.txt"});

        while (true)
        {
            std::string message;
            std::cout << "Введите запрос: " << '\n';
            std::getline(std::cin, message);
            std::cout << rag.request(message, {0}) + "\n";
        }
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
}
