using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;



namespace Perceptron
{
    class NeuralNetwork
    {
        List<List<int>> learningList;
        List<int> questionList;
        Net net;
        int currentLearningNumber;
        int currentEpocheNumber;
        public double currentError;

        static double alpha = 0.5;
        static double learningRate = 0.3f;
        static int countOfLayersInNet = 3;
        static int maxEpocheCount = 210;
        static double minError = 0.0001f;

        public List<double> errors;

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
            public double[][] neuronErrors;


            public Net(int countOfLayers)
            {
                layers = new Layer[countOfLayers];

                layers[0] = new Layer(100);
                layers[1] = new Layer(35);
                layers[2] = new Layer(1);

                weights = new double[countOfLayers][][];
                for (int i = 1; i < countOfLayers; i++)
                  {
                     weights[i] = new double[layers[i].neuronsOnLayer.Length][];
                     for (int j = 0; j < layers[i].neuronsOnLayer.Length; j++)
                         weights[i][j] = new double[layers[i-1].neuronsOnLayer.Length];
                  }


                Random random = new Random();
                for (int i = 1; i < countOfLayers; i++)
                    for (int j = 0; j < layers[i].neuronsOnLayer.Length; j++)
                        for (int k = 0; k < layers[i - 1].neuronsOnLayer.Length; k++)
                            weights[i][j][k] = random.NextDouble() * (0.4f - 0.1f) + 0.1f;

                neuronErrors = new double[countOfLayers][];
                for (int i = 0; i < countOfLayers; i++)
                    neuronErrors[i] = new double[layers[i].neuronsOnLayer.Length];
            }

        }

        public NeuralNetwork()
        {
            net = new Net(countOfLayersInNet);
            learningList = new List<List<int>>();
            currentLearningNumber = 0;
            currentEpocheNumber = 0;
            currentError = double.MaxValue;
            errors = new List<double>();   
        }

        public void initNetworkWithLearningList(List<List<int>> list)
        {
            learningList = list;
            initializeOutputs(list.ElementAt(0));
        }

        public void setQuestionList(List<int> qlist)
        {
            questionList = qlist;
        }

        void initializeOutputs(List<int> qList)
        {
            for (int i = 0; i < net.layers[0].neuronsOnLayer.Length; i++)
                net.layers[0].neuronsOnLayer[i].output = qList.ElementAt(i);
        }

        double getAnswerFromTeacherForLearningList(int number)
        {
            return learningList[number].ElementAt(100);
        }

        double activationFunction(double x)
        {
           return Math.Tanh(alpha*x);
        }

        double activationFunctionDerivative(double x)
        {
            double t = Math.Tanh(alpha * x);
            return alpha * (1 - t * t);
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
                currentError += Math.Pow(net.layers[countOfLayersInNet-1].neuronsOnLayer[i].output - learningList[currentLearningNumber].ElementAt(100),2);

            currentError /= 2;
            errors.Add(currentError);
        }

        void calculateNeuronErrors()
        {
            for (int l = countOfLayersInNet-1; l > 0; l--)
            {
                for (int i=0; i<net.layers[l].neuronsOnLayer.Length; i++)
                {
                    double tempValue=0;
                    if (l == (countOfLayersInNet - 1))
                    {
                        tempValue = (net.layers[l].neuronsOnLayer[i].output - getAnswerFromTeacherForLearningList(currentLearningNumber));
                        //tempValue *= activationFunctionDerivative(net.layers[l].neuronsOnLayer[i].state);
                        net.neuronErrors[l][i] = tempValue;
                    }
                    else
                    {
                        for (int j = 0; j < net.layers[l + 1].neuronsOnLayer.Length; j++)
                            tempValue += net.neuronErrors[l + 1][j] * net.weights[l + 1][j][i];
                       // tempValue *= activationFunctionDerivative(net.layers[l].neuronsOnLayer[i].state);
                        net.neuronErrors[l][i] = tempValue;
                    }
                }
            }

        }

        void calculateNewWeights()
        {
            for (int l = 1; l < countOfLayersInNet; l++)
            {
                for (int i = 0; i < net.layers[l].neuronsOnLayer.Length; i++)
                {
                    for (int j = 0; j < net.layers[l - 1].neuronsOnLayer.Length; j++)
                        net.weights[l][i][j] -= learningRate * net.neuronErrors[l][i] * net.layers[l - 1].neuronsOnLayer[j].output;                   
                }
            }
        }

        public void backwardPass()
        {
            calculateNeuronErrors();

            calculateNewWeights();            
        }

        public void trainNetwork()
        {
            while (/*currentError > minError &&*/ currentEpocheNumber < maxEpocheCount)
            {
                forwardPass();

                backwardPass();

                currentLearningNumber++;

                initializeOutputs(learningList[currentLearningNumber]);

                currentEpocheNumber++;
            }
        }

        public double askQuestion(List<int> question)
        {   
            initializeOutputs(question);
            forwardPass();
            double answer = net.layers[countOfLayersInNet - 1].neuronsOnLayer[net.layers[countOfLayersInNet-1].neuronsOnLayer.Length-1].output;
            return answer;
        }
    }
}
