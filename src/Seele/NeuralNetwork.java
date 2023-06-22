package Seele;

import Seele.ActivateFunction.ActivateFunction;
import Seele.LossFunction.LossFunction;
import Seele.Optimizer.*;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
public class NeuralNetwork {
    public List<Layer> layers;
    DataLoader dataset;
    LossFunction lossFunction;
    double learningRate;

    /**
     * Create a Neural Network, a neural network is formed by a bunch of layers
     */
    public NeuralNetwork(DataLoader dataset, double learningRate, LossFunction lossFunction) {
        layers = new ArrayList<>();
        this.dataset = dataset;
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    /**
     * Add a layer to the network
     * @param sizeOut The dimension of the next layer
     * @param optimizer The default optimizer is Adam
     */
    public void addLayer(int sizeOut, ActivateFunction activateFunction, Optimizer optimizer){
        Layer previousLayer = layers.isEmpty() ? null : layers.get(layers.size() - 1);
        int sizeIn = previousLayer == null ? dataset.getFeatures()[0].length : previousLayer.getSizeOut();
        layers.add(new Layer(sizeIn,sizeOut,activateFunction, optimizer));
    }
    public void addLayer(int sizeOut, ActivateFunction activateFunction){
        addLayer(sizeOut, activateFunction, new Adam());
    }

    /**
     * Train the network
     * @param interval After several epochs, print the loss
     */
    public void train(int numEpoch, OptionalInt interval){
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            double totalLoss = 0;
            for (int batchIndex = 0; batchIndex < dataset.batches.size(); batchIndex++){
                //Get data from dataloader
                DataLoader.Batch batch = dataset.batches.get(batchIndex);
                double[][] features = batch.getFeatures();
                double[][] targets = batch.getTargets();

                //training
                double[][] output = this.forward(features);
                double[][] gradient = lossFunction.gradient(output, targets);
                double loss = lossFunction.getLoss(output, targets);
                totalLoss += loss;

                this.backward(features, gradient);
            }

            //Print loss
            if (interval.isPresent() && epoch % interval.getAsInt() == 0){
                System.out.println("Epoch: " + (epoch+1) + " Loss: " + totalLoss/dataset.batches.size());
            }
        }
    }
    public void train(int numEpoch){
        train(numEpoch, OptionalInt.empty());
    }

    /**
     * After training finished, use this to predict something
     * @return The result of prediction
     */
    public double[][] predict(double[][] data) {
        return forward(data);
    }
    public double[][] predict(DataLoader data){
        return forward(data.getFeatures());
    }

    /**
     * The forward pass
     * @param input
     * @return The output
     */
    public double[][] forward(double[][] input){
        double[][] output = input;
        for (Layer layer:layers){
            output = layer.forward(output);
        }
        return output;
    }

    /**
     * The backward pass, unlike the forward pass, this process does not return anything,
     * it changes parameters automatically.
     */
    public void backward(double[][] features, double[][] gradient){
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            double[][] previousLayer = i == 0 ? features : layers.get(i - 1).outputAfterActivate;
            gradient = layer.backward(previousLayer, gradient, learningRate);
        }
    }
}
