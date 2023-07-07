#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define training_data_size (sizeof(training_data) / sizeof(training_data[0]))

typedef struct
{
  float w1;
  float w2;
  float b;
} Neuron;

typedef struct
{
  Neuron neurons[2];
  int num_neurons;
} Layer;

typedef struct
{
  Layer *layers;
  int num_layers;
} Network;

float training_data[4][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float feed_neuron(Neuron *neuron, float x1, float x2)
{
  float result = (x1 * neuron->w1) + (x2 * neuron->w2) + neuron->b;
  return sigmoid(result);
}

void feed_layer(Layer *layer, float x1, float x2, float *outputs)
{
  for (int i = 0; i < layer->num_neurons; i++)
  {
    outputs[i] = feed_neuron(&(layer->neurons[i]), x1, x2);
  }
}

float feed_forward(Network *network, float x1, float x2)
{
  float *inputs = malloc(2 * sizeof(float));
  inputs[0] = x1;
  inputs[1] = x2;

  for (int i = 0; i < network->num_layers; ++i)
  {
    float *outputs = malloc(2 * sizeof(float));
    feed_layer(&(network->layers[i]), inputs[0], inputs[1], outputs);
    free(inputs);
    inputs = outputs;
  }

  return inputs[0];
}

float random_number(void) { return ((float)rand() / (float)RAND_MAX); }

Network *setup_network()
{
  Network *network = malloc(sizeof(Network));

  network->layers = malloc(2 * sizeof(Layer));

  for (int i = 0; i < 2; i++)
  {
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

void free_network(Network *network)
{
  free(network->layers);
  free(network);
}

float cost(Network *network)
{
  float acc = 0.0f;
  for (int i = 0; i < training_data_size; ++i)
  {
    float x1 = training_data[i][0];
    float x2 = training_data[i][1];
    float y = feed_forward(network, x1, x2);
    float d = y - training_data[i][2];
    acc += d * d;
  }
  acc = acc / (float)training_data_size;
  return acc;
}

Network *finite_diff(Network *m, float h)
{
  Network *g = setup_network();

  float c = cost(m);

  for (int i = 0; i < m->num_layers; i++)
  {
    for (int j = 0; j < m->layers[i].num_neurons; j++)
    {
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

void learn(Network *m, Network *g, float learn_rate)
{
  for (int i = 0; i < m->num_layers; i++)
  {
    for (int j = 0; j < m->layers[i].num_neurons; j++)
    {
      m->layers[i].neurons[j].w1 -= learn_rate * g->layers[i].neurons[j].w1;
      m->layers[i].neurons[j].w2 -= learn_rate * g->layers[i].neurons[j].w2;
      m->layers[i].neurons[j].b -= learn_rate * g->layers[i].neurons[j].b;
    }
  }
}

void train(Network *m, int iterations, float h, float learn_rate)
{
  for (int i = 0; i < iterations; ++i)
  {
    Network *g = finite_diff(m, h);
    learn(m, g, learn_rate);
    free(g);
  }
}

void print_network(Network *network)
{
  for (int i = 0; i < network->num_layers; i++)
  {
    printf("Layer %d:\n", i + 1);
    for (int j = 0; j < network->layers[i].num_neurons; j++)
    {
      printf("  Neuron %d: w1 = %f, w2 = %f, b = %f\n", j + 1,
             network->layers[i].neurons[j].w1, network->layers[i].neurons[j].w2,
             network->layers[i].neurons[j].b);
    }
  }
}

int main()
{
  srand(time(NULL));

  Network *network = setup_network();

  printf("---------Pre Training----------\n");
  print_network(network);
  printf("\n\n");
  printf("Cost: %f\n", cost(network));

  train(network, 10000000, 1e-4, 1e-3);

  printf("---------Post Training----------\n");
  print_network(network);
  printf("\n\n");
  printf("Cost: %f\n", cost(network));

  printf("0 | 0 = %f\n", feed_forward(network, 0.0, 0.0));
  printf("0 | 1 = %f\n", feed_forward(network, 0.0, 1.0));
  printf("1 | 0 = %f\n", feed_forward(network, 1.0, 0.0));
  printf("1 | 1 = %f\n", feed_forward(network, 1.0, 1.0));
}