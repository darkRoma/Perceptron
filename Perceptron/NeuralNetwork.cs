using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;



namespace Perceptron
{
    class NeuralNetwork
    {
        public List<List<int>> learningList;
        public static int countOfLayersInNet = 3;
        Net net;
        public double currentError;

        public NeuralNetwork()
        {
            net = new Net(countOfLayersInNet);
            learningList = new List<List<int>>();
        }

        public void initializeOutputs()
        {
            for (int i = 0; i < net.layers[0].neuronsOnLayer.Length; i++)
                net.layers[0].neuronsOnLayer[i].output = learningList[0].ElementAt(i);
        }

        internal class Neuron
        {
            public double state;
            public double output;
            public Neuron()
            {
            }
        }

        internal class Layer
        {

            public Neuron[] neuronsOnLayer;
            int countOfNeurons;

            public Layer()
            {
                this.countOfNeurons = 1;
                neuronsOnLayer = new Neuron[countOfNeurons];
                neuronsOnLayer[0] = new Neuron();
            }

            public Layer(int countOfNeurons)
            {
                this.countOfNeurons = countOfNeurons;
                neuronsOnLayer = new Neuron[countOfNeurons];
                for (int i = 0; i < countOfNeurons; i++)
                    neuronsOnLayer[i] = new Neuron();
            }
        }

        internal class Net
        {
            public Layer[] layers;
            public double[][][] weights;
           
            public Net(int countOfLayers)
            {
                layers = new Layer[countOfLayers];

                layers[0] = new Layer(100);
                layers[1] = new Layer(10);
                layers[2] = new Layer(1);

                weights = new double[countOfLayers][][];
                for (int i = 1; i < countOfLayers; i++)
                  {
                     weights[i] = new double[layers[i].neuronsOnLayer.Length][];
                     for (int j = 0; j < layers[i].neuronsOnLayer.Length; j++)
                         weights[i][j] = new double[layers[i-1].neuronsOnLayer.Length];
                  }

                for (int i = 1; i < countOfLayers; i++)
                    for (int j = 0; j < layers[i].neuronsOnLayer.Length; j++)
                        for (int k = 0; k < layers[i-1].neuronsOnLayer.Length; k++)
                            weights[i][j][k] = 0.2f;

            }

        }

        double activationFunction(double x)
        {
            double alpha = 0.5;

            return Math.Tanh(alpha * x);
        }

        public void forwardPass()
        {
            for (int l = 1; l < countOfLayersInNet; l++) 
            {
                for (int i=0; i<net.layers[l].neuronsOnLayer.Length; i++)
                   {
                    double summator=0;

                    for (int j = 0; j < net.layers[l - 1].neuronsOnLayer.Length; j++)
                        summator += net.weights[l][i][j]*net.layers[l-1].neuronsOnLayer[j].output;

                    net.layers[l].neuronsOnLayer[i].state = summator;
                    net.layers[l].neuronsOnLayer[i].output = activationFunction(summator);
                   }
            }

            currentError = 0;
            for (int i = 0; i < net.layers[countOfLayersInNet-1].neuronsOnLayer.Length; i++)
                currentError += Math.Pow(net.layers[countOfLayersInNet-1].neuronsOnLayer[i].output - learningList[0].ElementAt(100),2);

            currentError /= 2;
        }
    }
}
