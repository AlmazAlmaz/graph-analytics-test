#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// Ребро в исходном CSV. Здесь id вершин еще не сжаты и совпадают с входными данными.
struct Edge {
    int32_t from = 0;
    int32_t to = 0;
};

// Параметры запуска. Часть параметров задается явно, остальные имеют рабочие значения
// по умолчанию, чтобы пример из examples/ запускался короткой командой.
struct Config {
    fs::path input_path;
    fs::path output_path = "pagerank.csv";
    fs::path work_dir;
    int iterations = 20;
    double damping = 0.85;
    double tolerance = 1e-8;
    size_t threads = std::max(1u, std::thread::hardware_concurrency());
    size_t buckets = 0;
};

// Информация о вершине после первого прохода по CSV.
// Хранится исходный id и число исходящих ребер, сами ребра здесь не лежат.
struct VertexInfo {
    int32_t id = 0;
    uint64_t out_degree = 0;
};

// Смещения входного файла. Если есть заголовок from,to, следующие проходы
// начинают чтение сразу после него и не пытаются парсить заголовок как ребро.
struct FileLayout {
    uint64_t file_size = 0;
    uint64_t data_start = 0;
};

// Ребро во временном бинарном формате. Используются плотные индексы 0..V-1,
// чтобы итерации PageRank работали с массивами, а не с исходными int32 id.
struct BinaryEdge {
    uint32_t from = 0;
    uint32_t to = 0;
};

static_assert(sizeof(BinaryEdge) == 8, "BinaryEdge должен состоять из двух uint32");

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::string trim_copy(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t' || value.front() == '\r')) {
        value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r')) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

// from_chars быстрее и предсказуемее, чем stoi: он не бросает исключения
// на каждой ошибке парсинга и не зависит от локали процесса.
bool parse_int32(std::string_view text, int32_t &result) {
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return false;
    }
    const char *begin = trimmed.data();
    const char *end = trimmed.data() + trimmed.size();
    int32_t parsed = 0;
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end) {
        return false;
    }
    result = parsed;
    return true;
}

// Читаем только первые две колонки CSV. Это позволяет запускаться и на файлах,
// где после from,to есть дополнительные поля, которые для PageRank не нужны.
std::optional<Edge> parse_edge_line(const std::string &line, uint64_t line_number) {
    if (line.empty()) {
        return std::nullopt;
    }

    const size_t first_comma = line.find(',');
    if (first_comma == std::string::npos) {
        fail("Некорректный CSV в строке " + std::to_string(line_number) + ": ожидается 'from,to'");
    }
    const size_t second_comma = line.find(',', first_comma + 1);
    const size_t to_length = second_comma == std::string::npos
                                 ? std::string::npos
                                 : second_comma - first_comma - 1;

    int32_t from = 0;
    int32_t to = 0;
    if (!parse_int32(std::string_view(line).substr(0, first_comma), from) ||
        !parse_int32(std::string_view(line).substr(first_comma + 1, to_length), to)) {
        fail("Некорректный id вершины в строке " + std::to_string(line_number) + ": '" + line + "'");
    }

    return Edge{from, to};
}

// Один раз определяем, есть ли заголовок. Это важно для больших файлов:
// последующие проходы начинают чтение сразу с data_start.
FileLayout inspect_csv_layout(const fs::path &input_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        fail("Не удалось открыть входной файл: " + input_path.string());
    }

    std::string first_line;
    if (!std::getline(input, first_line)) {
        fail("Входной файл пустой: " + input_path.string());
    }

    const std::string normalized = trim_copy(first_line);
    const uint64_t file_size = fs::file_size(input_path);
    const bool has_header = normalized == "from,to" || normalized.rfind("from,to,", 0) == 0;
    const std::streampos after_header = input.tellg();
    const uint64_t data_start = has_header
                                    ? (after_header == std::streampos(-1)
                                           ? file_size
                                           : static_cast<uint64_t>(after_header))
                                    : 0;

    return FileLayout{file_size, data_start};
}

// Собственный open-addressing индекс нужен для экономии памяти.
// std::unordered_map проще, но на больших графах дает большой overhead на каждый ключ.
class Int32Index {
public:
    explicit Int32Index(const std::vector<VertexInfo> &vertices) {
        size_t capacity = 1;
        while (capacity < vertices.size() * 2) {
            capacity <<= 1;
        }
        if (capacity == 0) {
            capacity = 1;
        }

        keys_.assign(capacity, 0);
        values_.assign(capacity, 0);
        used_.assign(capacity, 0);
        mask_ = capacity - 1;

        for (uint32_t i = 0; i < vertices.size(); ++i) {
            insert(vertices[i].id, i);
        }
    }

    uint32_t at(int32_t key) const {
        size_t slot = hash(key) & mask_;
        while (used_[slot] != 0) {
            if (keys_[slot] == key) {
                return values_[slot];
            }
            slot = (slot + 1) & mask_;
        }
        fail("Внутренняя ошибка: id вершины не найден в плотном индексе");
    }

private:
    // Простое перемешивание битов. Оно помогает не получать длинные цепочки
    // пробирования, если id вершин идут подряд или имеют общий шаблон.
    static size_t hash(int32_t key) {
        uint32_t x = static_cast<uint32_t>(key);
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return static_cast<size_t>(x);
    }

    void insert(int32_t key, uint32_t value) {
        size_t slot = hash(key) & mask_;
        while (used_[slot] != 0) {
            if (keys_[slot] == key) {
                values_[slot] = value;
                return;
            }
            slot = (slot + 1) & mask_;
        }
        used_[slot] = 1;
        keys_[slot] = key;
        values_[slot] = value;
    }

    std::vector<int32_t> keys_;
    std::vector<uint32_t> values_;
    std::vector<uint8_t> used_;
    size_t mask_ = 0;
};

// Таблица первого прохода. Она собирает множество вершин и out-degree,
// но не хранит список ребер, иначе решение снова стало бы O(E) по памяти.
class OutDegreeMap {
public:
    OutDegreeMap() {
        rehash(1024);
    }

    void add_vertex(int32_t key) {
        add(key, 0);
    }

    void increment_out_degree(int32_t key) {
        add(key, 1);
    }

    std::vector<VertexInfo> to_sorted_vertices() const {
        std::vector<VertexInfo> vertices;
        vertices.reserve(size_);
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (used_[i] != 0) {
                vertices.push_back(VertexInfo{keys_[i], values_[i]});
            }
        }
        std::sort(vertices.begin(), vertices.end(), [](const VertexInfo &lhs, const VertexInfo &rhs) {
            return lhs.id < rhs.id;
        });
        return vertices;
    }

private:
    static size_t hash(int32_t key) {
        uint32_t x = static_cast<uint32_t>(key);
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return static_cast<size_t>(x);
    }

    void add(int32_t key, uint64_t delta) {
        // Держим load factor ниже ~0.7, чтобы операции оставались близкими к O(1).
        if ((size_ + 1) * 10 >= keys_.size() * 7) {
            rehash(keys_.size() * 2);
        }

        size_t slot = hash(key) & mask_;
        while (used_[slot] != 0) {
            if (keys_[slot] == key) {
                values_[slot] += delta;
                return;
            }
            slot = (slot + 1) & mask_;
        }

        used_[slot] = 1;
        keys_[slot] = key;
        values_[slot] = delta;
        ++size_;
    }

    void rehash(size_t new_capacity) {
        size_t capacity = 1;
        while (capacity < new_capacity) {
            capacity <<= 1;
        }

        std::vector<int32_t> old_keys = std::move(keys_);
        std::vector<uint64_t> old_values = std::move(values_);
        std::vector<uint8_t> old_used = std::move(used_);

        keys_.assign(capacity, 0);
        values_.assign(capacity, 0);
        used_.assign(capacity, 0);
        mask_ = capacity - 1;
        size_ = 0;

        for (size_t i = 0; i < old_keys.size(); ++i) {
            if (old_used[i] != 0) {
                add(old_keys[i], old_values[i]);
            }
        }
    }

    std::vector<int32_t> keys_;
    std::vector<uint64_t> values_;
    std::vector<uint8_t> used_;
    size_t mask_ = 0;
    size_t size_ = 0;
};

std::vector<VertexInfo> collect_vertices(const fs::path &input_path, const FileLayout &layout) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        fail("Не удалось открыть входной файл: " + input_path.string());
    }
    input.seekg(static_cast<std::streamoff>(layout.data_start));

    // В первом проходе храним только вершины и out-degree, но не сами ребра.
    OutDegreeMap out_degrees;
    uint64_t edge_count = 0;

    std::string line;
    uint64_t line_number = layout.data_start == 0 ? 1 : 2;
    while (std::getline(input, line)) {
        const auto edge = parse_edge_line(line, line_number);
        ++line_number;
        if (!edge.has_value()) {
            continue;
        }

        out_degrees.increment_out_degree(edge->from);
        out_degrees.add_vertex(edge->to);
        ++edge_count;
    }

    if (edge_count == 0) {
        fail("Во входном графе нет ребер");
    }

    return out_degrees.to_sorted_vertices();
}

fs::path bucket_path(const fs::path &work_dir, size_t bucket) {
    return work_dir / ("bucket_" + std::to_string(bucket) + ".bin");
}

// Номер bucket вычисляется только по to-вершине. За счет этого каждый bucket
// отвечает за свой диапазон next_rank, а потоки не конфликтуют при записи.
size_t bucket_for_vertex(uint32_t vertex, size_t vertex_count, size_t bucket_count) {
    return static_cast<size_t>((static_cast<uint64_t>(vertex) * bucket_count) / vertex_count);
}

uint32_t ceil_div(size_t numerator, size_t denominator) {
    return static_cast<uint32_t>((numerator + denominator - 1) / denominator);
}

// Левая граница диапазона dense-id, который обслуживает bucket.
uint32_t bucket_begin(size_t bucket, size_t vertex_count, size_t bucket_count) {
    return ceil_div(vertex_count * bucket, bucket_count);
}

// Правая граница диапазона dense-id, который обслуживает bucket. Граница не включается.
uint32_t bucket_end(size_t bucket, size_t vertex_count, size_t bucket_count) {
    return ceil_div(vertex_count * (bucket + 1), bucket_count);
}

uint64_t build_edge_buckets(const Config &config,
                            const FileLayout &layout,
                            const std::vector<VertexInfo> &vertices,
                            const Int32Index &index) {
    // Второй проход по CSV: текстовые ребра преобразуются в компактные бинарные записи.
    // После этого итерации PageRank уже не тратят время на парсинг CSV.
    fs::create_directories(config.work_dir);

    std::vector<std::ofstream> buckets(config.buckets);
    for (size_t bucket = 0; bucket < config.buckets; ++bucket) {
        buckets[bucket].open(bucket_path(config.work_dir, bucket), std::ios::binary | std::ios::trunc);
        if (!buckets[bucket]) {
            fail("Не удалось создать bucket-файл: " + bucket_path(config.work_dir, bucket).string());
        }
    }

    std::ifstream input(config.input_path, std::ios::binary);
    if (!input) {
        fail("Не удалось открыть входной файл: " + config.input_path.string());
    }
    input.seekg(static_cast<std::streamoff>(layout.data_start));

    uint64_t edge_count = 0;
    std::string line;
    uint64_t line_number = layout.data_start == 0 ? 1 : 2;
    while (std::getline(input, line)) {
        const auto edge = parse_edge_line(line, line_number);
        ++line_number;
        if (!edge.has_value()) {
            continue;
        }

        // В bucket-файлы пишем уже плотные номера вершин, чтобы итерации не парсили CSV.
        const BinaryEdge binary_edge{index.at(edge->from), index.at(edge->to)};
        const size_t bucket = bucket_for_vertex(binary_edge.to, vertices.size(), config.buckets);
        buckets[bucket].write(reinterpret_cast<const char *>(&binary_edge), sizeof(BinaryEdge));
        if (!buckets[bucket]) {
            fail("Не удалось записать bucket-файл: " + bucket_path(config.work_dir, bucket).string());
        }
        ++edge_count;
    }

    for (auto &bucket: buckets) {
        bucket.close();
    }
    return edge_count;
}

void process_bucket(const Config &config,
                    size_t bucket,
                    const std::vector<VertexInfo> &vertices,
                    const std::vector<double> &rank,
                    std::vector<double> &next_rank,
                    double damping) {
    // Один bucket содержит только ребра в свой диапазон to-вершин.
    // Поэтому эта функция может писать в next_rank без mutex и atomic.
    const fs::path path = bucket_path(config.work_dir, bucket);
    const uint32_t begin = bucket_begin(bucket, vertices.size(), config.buckets);
    const uint32_t end = bucket_end(bucket, vertices.size(), config.buckets);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("Не удалось открыть bucket-файл: " + path.string());
    }

    BinaryEdge edge;
    while (input.read(reinterpret_cast<char *>(&edge), sizeof(BinaryEdge))) {
        // Защита от поврежденных временных файлов и ошибок в разбиении.
        // Если bucket содержит чужую to-вершину, результату доверять нельзя.
        if (edge.from >= vertices.size() || edge.to >= vertices.size() || edge.to < begin || edge.to >= end) {
            fail("Bucket-файл поврежден или ребро попало не в свой диапазон: " + path.string());
        }
        const uint64_t out_degree = vertices[edge.from].out_degree;
        next_rank[edge.to] += damping * rank[edge.from] / static_cast<double>(out_degree);
    }
    if (!input.eof()) {
        fail("Ошибка чтения bucket-файла: " + path.string());
    }
}

void process_buckets_parallel(const Config &config,
                              const std::vector<VertexInfo> &vertices,
                              const std::vector<double> &rank,
                              std::vector<double> &next_rank) {
    const size_t worker_count = std::min(config.threads, config.buckets);
    size_t next_bucket = 0;
    std::exception_ptr error;
    std::mutex mutex;

    // Потоки берут bucket-файлы из общей очереди. Mutex нужен только для выдачи
    // следующего bucket и сохранения исключения, не для самого PageRank-обновления.
    auto worker = [&]() {
        while (true) {
            size_t bucket = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (error != nullptr || next_bucket >= config.buckets) {
                    return;
                }
                bucket = next_bucket++;
            }

            try {
                process_bucket(config, bucket, vertices, rank, next_rank, config.damping);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (error == nullptr) {
                    error = std::current_exception();
                }
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers.emplace_back(worker);
    }
    for (auto &thread: workers) {
        thread.join();
    }
    if (error != nullptr) {
        std::rethrow_exception(error);
    }
}

void print_usage(const char *argv0) {
    std::cerr
            << "Использование: " << argv0 << " --input edges.csv --output pagerank.csv [опции]\n"
            << "\n"
            << "Опции:\n"
            << "  --iterations N    Максимальное число итераций PageRank (по умолчанию: 20)\n"
            << "  --damping X       Коэффициент damping в диапазоне [0, 1] (по умолчанию: 0.85)\n"
            << "  --tolerance X     Остановка, когда L1 delta меньше X (по умолчанию: 1e-8)\n"
            << "  --threads N       Число рабочих потоков (по умолчанию: hardware concurrency)\n"
            << "  --buckets N       Число bucket-файлов на диске (по умолчанию: 4 * threads)\n"
            << "  --work-dir PATH   Каталог для временных бинарных bucket-файлов\n";
}

Config parse_args(int argc, char **argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string &option) -> std::string {
            if (i + 1 >= argc) {
                fail("Не указано значение для " + option);
            }
            return argv[++i];
        };

        if (arg == "--input") {
            config.input_path = require_value(arg);
        } else if (arg == "--output") {
            config.output_path = require_value(arg);
        } else if (arg == "--work-dir") {
            config.work_dir = require_value(arg);
        } else if (arg == "--iterations") {
            config.iterations = std::stoi(require_value(arg));
        } else if (arg == "--damping") {
            config.damping = std::stod(require_value(arg));
        } else if (arg == "--tolerance") {
            config.tolerance = std::stod(require_value(arg));
        } else if (arg == "--threads") {
            config.threads = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--buckets") {
            config.buckets = static_cast<size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("Неизвестный аргумент: " + arg);
        }
    }

    if (config.input_path.empty()) {
        fail("Нужно указать --input");
    }
    if (config.iterations <= 0) {
        fail("--iterations должен быть положительным");
    }
    if (config.damping < 0.0 || config.damping > 1.0) {
        fail("--damping должен быть в диапазоне [0, 1]");
    }
    if (config.tolerance < 0.0) {
        fail("--tolerance не должен быть отрицательным");
    }
    config.threads = std::max<size_t>(1, config.threads);
    if (config.buckets == 0) {
        config.buckets = config.threads * 4;
    }
    config.buckets = std::max<size_t>(1, config.buckets);
    if (config.work_dir.empty()) {
        config.work_dir = config.output_path;
        config.work_dir += ".work";
    }

    return config;
}

void write_result(const fs::path &output_path,
                  const std::vector<VertexInfo> &vertices,
                  const std::vector<double> &rank) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        fail("Не удалось открыть выходной файл: " + output_path.string());
    }

    output << "vertex,rank\n";
    output << std::setprecision(12);
    for (size_t i = 0; i < vertices.size(); ++i) {
        output << vertices[i].id << ',' << rank[i] << '\n';
    }
}

int run(const Config &config) {
    const FileLayout layout = inspect_csv_layout(config.input_path);

    std::cerr << "Собираю вершины и исходящие степени...\n";
    const std::vector<VertexInfo> vertices = collect_vertices(config.input_path, layout);
    const size_t n = vertices.size();
    if (n > std::numeric_limits<uint32_t>::max()) {
        fail("Слишком много уникальных вершин для плотного uint32-индекса");
    }
    const uint64_t edge_count = std::accumulate(vertices.begin(), vertices.end(), uint64_t{0},
                                                [](uint64_t sum, const VertexInfo &vertex) {
                                                    return sum + vertex.out_degree;
                                                });

    std::cerr << "Вершин: " << n << ", ребер: " << edge_count << "\n";
    std::cerr << "Строю компактный индекс вершин...\n";
    // Плотный индекс нужен, чтобы rank хранить в vector<double>, а не в map по int32 id.
    const Int32Index index(vertices);

    // Ребра остаются на диске. В памяти после этого по-прежнему нет adjacency list.
    const uint64_t bucketed_edges = build_edge_buckets(config, layout, vertices, index);
    if (bucketed_edges != edge_count) {
        fail("Внутренняя ошибка: число ребер изменилось при построении bucket-файлов");
    }
    std::cerr << "Bucket-файлов на диске: " << config.buckets << ", каталог: " << config.work_dir << "\n";

    std::vector<double> rank(n, 1.0 / static_cast<double>(n));
    std::vector<double> next_rank(n, 0.0);

    for (int iteration = 1; iteration <= config.iterations; ++iteration) {
        // Вершины без исходящих ребер не передают rank по edge list.
        // Их масса добавляется в base и равномерно распределяется по всем вершинам.
        const double dangling_sum = [&]() {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (vertices[i].out_degree == 0) {
                    sum += rank[i];
                }
            }
            return sum;
        }();

        // base - общий начальный вклад каждой вершины: teleportation плюс dangling mass.
        // Дальше к нему добавляются вклады реальных ребер из bucket-файлов.
        const double base = (1.0 - config.damping) / static_cast<double>(n) +
                            config.damping * dangling_sum / static_cast<double>(n);
        std::fill(next_rank.begin(), next_rank.end(), base);

        // Bucket-и разбиты по to-вершинам, поэтому потоки пишут в разные диапазоны next_rank.
        process_buckets_parallel(config, vertices, rank, next_rank);

        double rank_sum = 0.0;
        for (const double value: next_rank) {
            rank_sum += value;
        }

        // Нормализация убирает небольшой накопленный дрейф из-за операций с double.
        if (rank_sum > 0.0) {
            for (double &value: next_rank) {
                value /= rank_sum;
            }
        }

        // L1 delta показывает, насколько изменился весь вектор rank за итерацию.
        // Когда delta мала, дальнейшие итерации почти не меняют распределение.
        double delta = 0.0;
        for (size_t i = 0; i < n; ++i) {
            delta += std::abs(next_rank[i] - rank[i]);
        }

        rank.swap(next_rank);
        std::cerr << "Итерация " << iteration << ": L1 delta = " << std::setprecision(8) << delta << "\n";
        if (delta < config.tolerance) {
            std::cerr << "Остановлено по tolerance.\n";
            break;
        }
    }

    const double final_sum = std::accumulate(rank.begin(), rank.end(), 0.0);
    const double min_rank = *std::min_element(rank.begin(), rank.end());
    std::cerr << "Итоговая сумма rank = " << std::setprecision(12) << final_sum
            << ", минимальный rank = " << min_rank << "\n";

    write_result(config.output_path, vertices, rank);
    std::cerr << "Результат записан: " << config.output_path << "\n";
    return 0;
}

int main(int argc, char **argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception &ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
}
