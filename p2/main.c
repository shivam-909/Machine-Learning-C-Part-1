
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float training_data[30][3] = {
    {0.1, 0.2, 0.9},  {0.3, 0.4, 2.1},  {0.5, 0.6, 3.3},  {0.7, 0.8, 4.5},
    {0.9, 1.0, 5.7},  {1.1, 1.2, 6.9},  {1.3, 1.4, 8.1},  {1.5, 1.6, 9.3},
    {1.7, 1.8, 10.5}, {1.9, 2.0, 11.7}, {2.1, 2.2, 12.9}, {2.3, 2.4, 14.1},
    {2.5, 2.6, 15.3}, {2.7, 2.8, 16.5}, {2.9, 3.0, 17.7}, {3.1, 3.2, 18.9},
    {3.3, 3.4, 20.1}, {3.5, 3.6, 21.3}, {3.7, 3.8, 22.5}, {3.9, 4.0, 23.7},
    {4.1, 4.2, 24.9}, {4.3, 4.4, 26.1}, {4.5, 4.6, 27.3}, {4.7, 4.8, 28.5},
    {4.9, 5.0, 29.7}, {5.1, 5.2, 30.9}, {5.3, 5.4, 32.1}, {5.5, 5.6, 33.3},
    {5.7, 5.8, 34.5}, {5.9, 6.0, 35.7}};

#define training_data_size (sizeof(training_data) / sizeof(training_data[0]))

float random_number(void) { return ((float)rand() / (float)RAND_MAX); }

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float relu(float x) { return x > 0 ? x : 0; }

typedef struct {
  float w1;
  float w2;
  float b;
} Neuron;

typedef struct {
  Neuron neurons[2];
  int num_neurons;
} Layer;

typedef struct {
  Layer *layers;
  int num_layers;
} Network;

float feed_neuron(Neuron *neuron, float x1, float x2) {
  float result = (x1 * neuron->w1) + (x2 * neuron->w2) + neuron->b;
  return result;
}

void feed_layer(Layer *layer, float x1, float x2, float *outputs) {
  for (int i = 0; i < layer->num_neurons; i++) {
    outputs[i] = feed_neuron(&(layer->neurons[i]), x1, x2);
  }
}

float feed_forward(Network *network, float x1, float x2) {
  float *inputs = malloc(2 * sizeof(float));
  inputs[0] = x1;
  inputs[1] = x2;

  for (int i = 0; i < network->num_layers; ++i) {
    float *outputs = malloc(2 * sizeof(float));
    feed_layer(&(network->layers[i]), inputs[0], inputs[1], outputs);
    free(inputs);
    inputs = outputs;
  }

  return inputs[0];
}

Network *setup_network() {
  Network *network = malloc(sizeof(Network));

  network->layers = malloc(2 * sizeof(Layer));

  for (int i = 0; i < 2; i++) {
    network->layers[0].neurons[i].w1 = random_number();
    network->layers[0].neurons[i].w2 = random_number();
    network->layers[0].neurons[i].b = random_number();
  }

  network->layers[0].num_neurons = 2;

  network->layers[1].neurons[0].w1 = random_number();
  network->layers[1].neurons[0].w2 = random_number();
  network->layers[1].neurons[0].b = random_number();

  network->layers[1].num_neurons = 1;

  network->num_layers = 2;

  return network;
}

void free_network(Network *network) {
  free(network->layers);
  free(network);
}

float cost(Network *network) {
  float acc = 0.0f;
  for (int i = 0; i < training_data_size; ++i) {
    float x1 = training_data[i][0];
    float x2 = training_data[i][1];
    float y = feed_forward(network, x1, x2);
    float d = y - training_data[i][2];
    acc += d * d;
  }
  acc = acc / (float)training_data_size;
  return acc;
}

Network *finite_diff(Network *m, float h) {
  Network *g = setup_network();

  float c = cost(m);

  for (int i = 0; i < m->num_layers; i++) {
    for (int j = 0; j < m->layers[i].num_neurons; j++) {
      float saved;
      saved = m->layers[i].neurons[j].w1;
      m->layers[i].neurons[j].w1 += h;
      g->layers[i].neurons[j].w1 = (cost(m) - c) / h;
      m->layers[i].neurons[j].w1 = saved;

      saved = m->layers[i].neurons[j].w2;
      m->layers[i].neurons[j].w2 += h;
      g->layers[i].neurons[j].w2 = (cost(m) - c) / h;
      m->layers[i].neurons[j].w2 = saved;

      saved = m->layers[i].neurons[j].b;
      m->layers[i].neurons[j].b += h;
      g->layers[i].neurons[j].b = (cost(m) - c) / h;
      m->layers[i].neurons[j].b = saved;
    }
  }

  return g;
}

void learn(Network *m, Network *g, float learn_rate) {
  for (int i = 0; i < m->num_layers; i++) {
    for (int j = 0; j < m->layers[i].num_neurons; j++) {
      m->layers[i].neurons[j].w1 -= learn_rate * g->layers[i].neurons[j].w1;
      m->layers[i].neurons[j].w2 -= learn_rate * g->layers[i].neurons[j].w2;
      m->layers[i].neurons[j].b -= learn_rate * g->layers[i].neurons[j].b;
    }
  }
}

void train(Network *m, int iterations, float h, float learn_rate) {
  for (int i = 0; i < iterations; ++i) {
    Network *g = finite_diff(m, h);
    learn(m, g, learn_rate);
    free(g);
  }
}

void print_network(Network *network) {
  for (int i = 0; i < network->num_layers; i++) {
    printf("Layer %d:\n", i + 1);
    for (int j = 0; j < network->layers[i].num_neurons; j++) {
      printf("  Neuron %d: w1 = %f, w2 = %f, b = %f\n", j + 1,
             network->layers[i].neurons[j].w1, network->layers[i].neurons[j].w2,
             network->layers[i].neurons[j].b);
    }
  }
}

int main() {
  srand(time(NULL));

  Network *network = setup_network();

  printf("---------Pre Training----------\n");
  print_network(network);
  printf("\n\n");
  printf("Cost: %f\n", cost(network));

  train(network, 1000000, 1e-3, 1e-5);

  printf("---------Post Training----------\n");
  print_network(network);
  printf("\n\n");
  printf("Cost: %f\n", cost(network));

  printf("4 and 5 = %f\n", feed_forward(network, 4.0, 5.0));
  printf("6 and 8 = %f\n", feed_forward(network, 6.0, 8.0));
  printf("10 and 12 = %f\n", feed_forward(network, 10.0, 12.0));
  printf("15 and 21 = %f\n", feed_forward(network, 15.0, 21.0));
}
