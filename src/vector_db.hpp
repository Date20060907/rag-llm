#ifndef VECTOR_DB_H
#define VECTOR_DB_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

struct VectorRecord {
    uint32_t id;
    std::vector<float> embedding;
    std::string metadata;
};

class VectorDatabase {
private:
    std::string filename;
    size_t dimension;
    std::vector<VectorRecord> vectors;
    bool modified;
    std::unordered_map<uint32_t, size_t> id_to_index;

public:
    VectorDatabase(const std::string& db_filename, size_t dim);
    ~VectorDatabase();

    bool initialize();
    uint32_t addEmbedding(const std::vector<float>& embedding, const std::string& metadata = "");

    std::vector<std::pair<uint32_t, float>> findTopK(
        const std::vector<float>& query,
        uint32_t k = 5,
        float similarity_threshold = 0.0f);

    bool save();
    bool load();
    size_t size() const { return vectors.size(); }

    std::string getMetadata(uint32_t id) const;
    bool updateMetadata(uint32_t id, const std::string& new_metadata);

private:
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const;
    void normalizeVector(std::vector<float>& vector) const;
    uint32_t generateId() const;
};

#endif
