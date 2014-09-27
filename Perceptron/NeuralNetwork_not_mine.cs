using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Perceptron
{
    class NeuralNetwork_not_mine
    {
        internal class HalfSquaredEuclidianDistance
        {
            public double calculateError(double[] v1, double[] v2)
            {
                double d = 0;
                for (int i = 0; i < v1.Length; i++)
                {
                    d += (v1[i] - v2[i]) * (v1[i] - v2[i]);
                }
                return 0.5 * d;
            }

            public double calculatePartialDerivaitveByV2Index(double[] v1, double[] v2, int v2Index)
            {
                return v2[v2Index] - v1[v2Index];
            }

        }

        internal class HyperbolicTangensFunction
        {

            private double _alpha = 1;

            internal HyperbolicTangensFunction(double alpha)
            {
                _alpha = alpha;
            }

            public double Compute(double x)
            {
                return (Math.Tanh(_alpha * x));
            }

            public double ComputeFirstDerivative(double x)
            {
                double t = Math.Tanh(_alpha * x);
                return _alpha * (1 - t * t);
            }
        }

        public interface INeuron
        {

            /// <summary>
            /// Weights of the neuron
            /// </summary>
            double[] Weights { get; }

            /// <summary>
            /// Offset/bias of neuron (default is 0)
            /// </summary>
            double Bias { get; set; }

            /// <summary>
            /// Compute NET of the neuron by input vector
            /// </summary>
            /// <param name="inputVector">Input vector (must be the same dimension as was set in SetDimension)</param>
            /// <returns>NET of neuron</returns>
            double NET(double[] inputVector);

            /// <summary>
            /// Compute state of neuron
            /// </summary>
            /// <param name="inputVector">Input vector (must be the same dimension as was set in SetDimension)</param>
            /// <returns>State of neuron</returns>
            double Activate(double[] inputVector);

            /// <summary>
            /// Last calculated state in Activate
            /// </summary>
            double LastState { get; set; }

            /// <summary>
            /// Last calculated NET in NET
            /// </summary>
            double LastNET { get; set; }

            IList<INeuron> Childs { get; }

            IList<INeuron> Parents { get; }

            HyperbolicTangensFunction ActivationFunction { get; set; }

            double dEdz { get; set; }
        }

        public interface ILayer
        {

            /// <summary>
            /// Compute output of the layer
            /// </summary>
            /// <param name="inputVector">Input vector</param>
            /// <returns>Output vector</returns>
            double[] Compute(double[] inputVector);

            /// <summary>
            /// Get last output of the layer
            /// </summary>
            double[] LastOutput { get; }

            /// <summary>
            /// Get neurons of the layer
            /// </summary>
            INeuron[] Neurons { get; }

            /// <summary>
            /// Get input dimension of neurons
            /// </summary>
            int InputDimension { get; }
        }

        public interface INeuralNetwork
        {

            /// <summary>
            /// Compute output vector by input vector
            /// </summary>
            /// <param name="inputVector">Input vector (double[])</param>
            /// <returns>Output vector (double[])</returns>
            double[] ComputeOutput(double[] inputVector);

            //Stream Save();

            /// <summary>
            /// Train network with given inputs and outputs
            /// </summary>
            /// <param name="inputs">Set of input vectors</param>
            /// <param name="outputs">Set if output vectors</param>
            void Train();
        }

        public interface IMultilayerNeuralNetwork : INeuralNetwork
        {
            /// <summary>
            /// Get array of layers of network
            /// </summary>
            ILayer[] Layers { get; }
        }

        public class LearningAlgorithmConfig
        {

            public double LearningRate { get; set; }

            /// <summary>
            /// Size of the butch. -1 means fullbutch size. 
            /// </summary>
            public int BatchSize { get; set; }

            public double RegularizationFactor { get; set; }

            public int MaxEpoches { get; set; }

            /// <summary>
            /// If cumulative error for all training examples is less then MinError, then algorithm stops 
            /// </summary>
            public double MinError { get; set; }

            /// <summary>
            /// If cumulative error change for all training examples is less then MinErrorChange, then algorithm stops 
            /// </summary>
            public double MinErrorChange { get; set; }

            /// <summary>
            /// Function to minimize
            /// </summary>
            //public IMetrics<double> ErrorFunction { get; set; }

        }

        public class DataItem<T>
        {
            private T[] _input = null;
            private T[] _output = null;

            public DataItem()
            {
            }

            public DataItem(T[] input, T[] output)
            {
                _input = input;
                _output = output;
            }

            public T[] Input
            {
                get { return _input; }
                set { _input = value; }
            }

            public T[] Output
            {
                get { return _output; }
                set { _output = value; }
            }
        }

        public void train()
        {

            LearningAlgorithmConfig _config = new LearningAlgorithmConfig();

            int someTempCount = 100;

            if (_config.BatchSize < 1 || _config.BatchSize > someTempCount)
            {
                _config.BatchSize = someTempCount;
            }
            double currentError = Single.MaxValue;
            double lastError = 0;
            int epochNumber = 0;

            do 
            {

                lastError = currentError;
                DateTime dtStart = DateTime.Now;

                //preparation for epoche
                int[] trainingIndices = new int[someTempCount];
                for (int i = 0; i < someTempCount; i++)
                {
                    trainingIndices[i] = i;
                }
                if (_config.BatchSize > 0)
                {
                    //trainingIndices = Shuffle(trainingIndices);
                    Random rnd = new Random();
                    trainingIndices = trainingIndices.OrderBy(x => rnd.Next()).ToArray();
                }


                IMultilayerNeuralNetwork network = null;
                //DataItem<double> data;

                //process data set
                int currentIndex = 0;
                do
                {


                    #region initialize accumulated error for batch, for weights and biases

                    

                    double[][][] nablaWeights = new double[network.Layers.Length][][];
                    double[][] nablaBiases = new double[network.Layers.Length][];

                    for (int i = 0; i < network.Layers.Length; i++)
                    {
                        nablaBiases[i] = new double[network.Layers[i].Neurons.Length];
                        nablaWeights[i] = new double[network.Layers[i].Neurons.Length][];
                        for (int j = 0; j < network.Layers[i].Neurons.Length; j++)
                        {
                            nablaBiases[i][j] = 0;
                            nablaWeights[i][j] = new double[network.Layers[i].Neurons[j].Weights.Length];
                            for (int k = 0; k < network.Layers[i].Neurons[j].Weights.Length; k++)
                            {
                                nablaWeights[i][j][k] = 0;
                            }
                        }
                    }

                    #endregion

                    //process one batch
                    for (int inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < someTempCount; inBatchIndex++)
                    {
                        //forward pass
                      //  double[] realOutput = network.ComputeOutput(data[trainingIndices[inBatchIndex]].Input);

                        //backward pass, error propagation
                        //last layer
                        //.......................................ОБРАБОТКА ПОСЛЕДНЕГО СЛОЯ

                        //hidden layers
                        //.......................................ОБРАБОТКА СКРЫТЫХ СЛОЕВ
                    }

                    //update weights and bias
                    for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
                    {
                        for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
                        {
                            network.Layers[layerIndex].Neurons[neuronIndex].Bias -= nablaBiases[layerIndex][neuronIndex];
                            for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
                            {
                                network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] -=
                                    nablaWeights[layerIndex][neuronIndex][weightIndex];
                            }
                        }
                    }

                    currentIndex += _config.BatchSize;
                } while (currentIndex < someTempCount);


            } while (epochNumber < _config.MaxEpoches &&
                     currentError > _config.MinError &&
                     Math.Abs(currentError - lastError) > _config.MinErrorChange);


        }

    }
}
