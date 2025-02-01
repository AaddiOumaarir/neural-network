#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <random>

using namespace std;

// Fonction pour inverser l'endianness des entiers 32 bits
uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
}

// Lecture des images MNIST
vector<vector<double>> read_mnist_images(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        cerr << "Invalid MNIST image file!" << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = swap_endian(num_images);
    file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    vector<vector<double>> images(num_images, vector<double>(rows * cols));
    for (size_t i = 0; i < num_images; ++i) {
        vector<unsigned char> buffer(rows * cols);
        file.read(reinterpret_cast<char*>(buffer.data()), rows * cols);
        for (size_t j = 0; j < rows * cols; ++j) {
            images[i][j] = buffer[j] / 255.0;
        }
    }

    return images;
}

// Lecture des labels MNIST
vector<int> read_mnist_labels(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        cerr << "Invalid MNIST label file!" << endl;
        exit(1);
    }

    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);

    vector<int> labels(num_labels);
    vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);
    for (size_t i = 0; i < num_labels; ++i) {
        labels[i] = static_cast<int>(buffer[i]);
    }

    return labels;
}

// Conversion des labels en one-hot encoding
vector<vector<double>> one_hot_encode(const vector<int>& labels, int num_classes = 10) {
    vector<vector<double>> encoded(labels.size(), vector<double>(num_classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        encoded[i][labels[i]] = 1.0;
    }
    return encoded;
}

class NeuralNetwork {
private:
    vector<vector<double>> weights1;
    vector<vector<double>> weights2;
    vector<double> bias1;
    vector<double> bias2;
    double learning_rate;

    mt19937 gen;
    normal_distribution<double> dist;

    static vector<double> softmax(const vector<double>& x) {
        vector<double> res(x.size());
        double max_x = *max_element(x.begin(), x.end());
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            res[i] = exp(x[i] - max_x);
            sum += res[i];
        }
        for (size_t i = 0; i < x.size(); ++i) {
            res[i] /= sum;
        }
        return res;
    }

public:
    NeuralNetwork(double lr) : gen(random_device{}()), dist(0.0, 0.1), learning_rate(lr) {
        // Initialisation des poids
        weights1.resize(784, vector<double>(128));
        for (auto& row : weights1) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        bias1.resize(128, 0.0);

        weights2.resize(128, vector<double>(10));
        for (auto& row : weights2) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
        bias2.resize(10, 0.0);
    }

    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            int correct = 0;

            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward propagation
                vector<double> z1(128, 0.0);
                vector<double> a1(128, 0.0);
                for (size_t j = 0; j < 128; ++j) {
                    for (size_t k = 0; k < 784; ++k) {
                        z1[j] += inputs[i][k] * weights1[k][j];
                    }
                    z1[j] += bias1[j];
                    a1[j] = max(0.0, z1[j]);
                }

                vector<double> z2(10, 0.0);
                for (size_t j = 0; j < 10; ++j) {
                    for (size_t k = 0; k < 128; ++k) {
                        z2[j] += a1[k] * weights2[k][j];
                    }
                    z2[j] += bias2[j];
                }
                vector<double> a2 = softmax(z2);

                // Calcul de la perte et précision
                double loss = 0.0;
                for (size_t j = 0; j < 10; ++j) {
                    loss += targets[i][j] * log(a2[j] + 1e-15);
                }
                total_loss -= loss;

                int predicted = max_element(a2.begin(), a2.end()) - a2.begin();
                int actual = max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();
                if (predicted == actual) correct++;

                // Backward propagation
                vector<double> delta2 = a2;
                for (size_t j = 0; j < 10; ++j) {
                    delta2[j] -= targets[i][j];
                }

                // Mise à jour des poids de la couche de sortie
                for (size_t k = 0; k < 128; ++k) {
                    for (size_t j = 0; j < 10; ++j) {
                        weights2[k][j] -= learning_rate * a1[k] * delta2[j];
                    }
                }
                for (size_t j = 0; j < 10; ++j) {
                    bias2[j] -= learning_rate * delta2[j];
                }

                // Calcul de l'erreur pour la couche cachée
                vector<double> delta1(128, 0.0);
                for (size_t k = 0; k < 128; ++k) {
                    double error = 0.0;
                    for (size_t j = 0; j < 10; ++j) {
                        error += weights2[k][j] * delta2[j];
                    }
                    delta1[k] = error * (z1[k] > 0 ? 1.0 : 0.0);
                }

                // Mise à jour des poids de la couche cachée
                for (size_t k = 0; k < 784; ++k) {
                    for (size_t j = 0; j < 128; ++j) {
                        weights1[k][j] -= learning_rate * inputs[i][k] * delta1[j];
                    }
                }
                for (size_t j = 0; j < 128; ++j) {
                    bias1[j] -= learning_rate * delta1[j];
                }
            }

            cout << "Epoch " << epoch + 1 << "/" << epochs
                 << " - Loss: " << total_loss / inputs.size()
                 << " - Accuracy: " << static_cast<double>(correct) / inputs.size() << endl;
        }
    }

    vector<double> predict(const vector<double>& input) {
        vector<double> z1(128, 0.0);
        vector<double> a1(128, 0.0);
        for (size_t j = 0; j < 128; ++j) {
            for (size_t k = 0; k < 784; ++k) {
                z1[j] += input[k] * weights1[k][j];
            }
            z1[j] += bias1[j];
            a1[j] = max(0.0, z1[j]);
        }

        vector<double> z2(10, 0.0);
        for (size_t j = 0; j < 10; ++j) {
            for (size_t k = 0; k < 128; ++k) {
                z2[j] += a1[k] * weights2[k][j];
            }
            z2[j] += bias2[j];
        }
        return softmax(z2);
    }
};

int main() {
    // Chargement des données
    auto train_images = read_mnist_images("train-images.idx3-ubyte");
    auto train_labels = read_mnist_labels("train-labels.idx1-ubyte");
    auto test_images = read_mnist_images("t10k-images.idx3-ubyte");
    auto test_labels = read_mnist_labels("t10k-labels.idx1-ubyte");

    auto train_targets = one_hot_encode(train_labels);
    auto test_targets = one_hot_encode(test_labels);

    // Création du réseau
    NeuralNetwork nn(0.01);
    
    // Entraînement
    nn.train(train_images, train_targets, 5);

    // Test
    int correct = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        auto output = nn.predict(test_images[i]);
        int predicted = max_element(output.begin(), output.end()) - output.begin();
        if (predicted == test_labels[i]) correct++;
    }
    cout << "Test Accuracy: " << static_cast<double>(correct) / test_images.size() << endl;

    return 0;
}