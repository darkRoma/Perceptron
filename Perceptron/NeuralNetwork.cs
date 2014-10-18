using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;



namespace Perceptron
{
    class NeuralNetwork
    {
        List<List<int>> learningList;
        List<List<int>> testingList;

        List<int> questionList;
        Net net;
        int currentLearningNumber;
        int currentEpocheNumber;
        double currentError;
        public double networkPower;


        public int epocheCount
        {
            get { return currentEpocheNumber; }
            set { epocheCount = value; }
        }

        //Config
        static double alpha = 1;
        static double learningRate = 0.5;
        static int countOfLayersInNet = 3;
        static int maxEpocheCount = 100;
        static double minError = 0.1f;
        static double minTestingErrorChange = 0.001f;
        static int percentOfLearningVectors = 70;

        public List<double> errors;
        public List<double> averageLearningErrorList;
        public List<double> averageTrainingErrorList;

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
                layers[1] = new Layer(30);
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
                            weights[i][j][k] = random.NextDouble() - 0.5f;


                neuronErrors = new double[countOfLayers][];
                for (int i = 0; i < countOfLayers; i++)
                    neuronErrors[i] = new double[layers[i].neuronsOnLayer.Length];
            }
        }

        public NeuralNetwork()
        {
            net = new Net(countOfLayersInNet);
            learningList = new List<List<int>>();
            testingList = new List<List<int>>();
            currentLearningNumber = 0;
            currentEpocheNumber = 0;
            currentError = double.MaxValue;
            errors = new List<double>();   
            averageLearningErrorList = new List<double>();
            averageTrainingErrorList = new List<double>();
            networkPower = 0;
        }

        public void initNetworkWithLearningList(List<List<int>> list)
        {
            learningList.Clear();
            testingList.Clear();

            double percentOfLearningVectorsDouble = (double)percentOfLearningVectors / 100;

            for (int i = 0; i < (list.Count * percentOfLearningVectorsDouble); i++)
                learningList.Add(list[i]);

            for (int i = (int)(list.Count * percentOfLearningVectorsDouble) + 1; i < list.Count; i++)
                testingList.Add(list[i]);
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

        public void forwardPass(bool isAnswer, double answer)
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

            if (isAnswer)
            {
                currentError = 0;
                for (int i = 0; i < net.layers[countOfLayersInNet - 1].neuronsOnLayer.Length; i++)
                    //currentError += Math.Pow(net.layers[countOfLayersInNet-1].neuronsOnLayer[i].output - learningList[currentLearningNumber].ElementAt(100),2);
                    currentError += Math.Pow(net.layers[countOfLayersInNet - 1].neuronsOnLayer[i].output - answer, 2);
             
                currentError /= 2;
                errors.Add(currentError);
            }            
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
                        //tempValue *= activationFunctionDerivative(net.layers[l].neuronsOnLayer[i].state);
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
            averageLearningErrorList.Add(1);
            averageTrainingErrorList.Add(2);
            int maxLearningVectorCount = learningList.Count-1;
            int maxTestingVectorCount = testingList.Count-1;

            List<double> tempList = new List<double>();
            bool noPikeEffect = true;
            while (currentEpocheNumber < maxEpocheCount && noPikeEffect)
            {
                // Training network with learningList
                
                for (currentLearningNumber = 0; currentLearningNumber < maxLearningVectorCount; currentLearningNumber++)
                {
                    initializeOutputs(learningList[currentLearningNumber]);

                    forwardPass(true, learningList[currentLearningNumber].ElementAt(100));

                    backwardPass();
                }

                double averageError = 0;
                for (int i = 0; i < errors.Count; i++)
                    averageError += errors[i];
                averageError /= errors.Count;
                averageLearningErrorList.Add(averageError);
                errors.Clear();

                //Testing network with testingList

                for (int currentTestingNumber = 0; currentTestingNumber < maxTestingVectorCount; currentTestingNumber++)
                {
                    initializeOutputs(testingList[currentTestingNumber]);

                    forwardPass(true, testingList[currentTestingNumber].ElementAt(100));
                }

                averageError = 0;
                for (int i = 0; i < errors.Count; i++)
                    averageError += errors[i];
                averageError /= errors.Count;
                averageTrainingErrorList.Add(averageError);
                errors.Clear();

                noPikeEffect = isPikeEffect();

                currentEpocheNumber++;
            }
            networkPower = (averageLearningErrorList.Last() + averageTrainingErrorList.Last()) / 2;
        }

        bool isPikeEffect()
        {
            bool errorLessThanMinError = averageTrainingErrorList.Last() < minError;
            bool errorGrowing = averageTrainingErrorList.Last() - averageTrainingErrorList.ElementAt(averageTrainingErrorList.Count - 2) > 0;
            bool errorStays = Math.Abs(averageTrainingErrorList.Last() - averageTrainingErrorList.ElementAt(averageTrainingErrorList.Count - 2)) < minTestingErrorChange;

            if ((errorLessThanMinError && errorGrowing) || errorStays) return false;

            return true;
        }

        public double askQuestion(List<int> question)
        {   
            initializeOutputs(question);
            forwardPass(false, 0);
            double answer = net.layers[countOfLayersInNet - 1].neuronsOnLayer[net.layers[countOfLayersInNet-1].neuronsOnLayer.Length-1].output;
            return answer;
        }
    }
}
