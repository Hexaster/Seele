package Seele;

import Seele.ActivateFunction.ActivateFunction;
import Seele.Optimizer.Optimizer;
import java.util.Random;

public class Layer {
    int sizeIn;
    int sizeOut;
    private double[][] weight;
    private double[] bias;
    private final ActivateFunction activateFunction;
    public double[][] outputBeforeActivate;
    public double[][] outputAfterActivate;
    private final Optimizer optimizer;
    Random random = new Random();
    public Layer(int sizeIn, int sizeOut, ActivateFunction activateFunction, Optimizer optimizer) {
        this.sizeIn = sizeIn;
        this.sizeOut = sizeOut;
        this.weight = new double[sizeIn][sizeOut];
        this.bias = new double[sizeOut];
        this.activateFunction = activateFunction;
        this.optimizer = optimizer;
        for (int i = 0; i < sizeIn; i++){
            for (int j = 0; j < sizeOut; j++){
                weight[i][j] = random.nextGaussian()*0.01;
            }
        }
        for (int j = 0; j < sizeOut; j++){
            bias[j] = random.nextGaussian()*0.01;
        }
    }

    /**
     * The forward pass of a layer
     * @param input
     * @return The result of activate function
     */
    public double[][] forward(double[][] input){
        outputBeforeActivate = Arithmetic.dot(input,weight);
        outputAfterActivate = new double[outputBeforeActivate.length][sizeOut];
        for (int i = 0; i < outputBeforeActivate.length; i++) {
            for (int j = 0; j < outputBeforeActivate[0].length; j++) {
                outputBeforeActivate[i][j] += bias[j];
            }
        }
        outputAfterActivate = activateFunction.actForward(outputBeforeActivate);
        return outputAfterActivate;
    }

    /**
     * The backward pass of a layer
     * @param lastLayer The layer of the last layer
     * @param gradient
     * @param learningRate
     * @return The derivative for the former layer
     */
    public double[][] backward(double[][] lastLayer, double[][] gradient, double learningRate) {
        double[][] gradientActivate = activateFunction.actBackward(outputBeforeActivate);
        for (int i = 0; i < gradient.length; i++) {
            for (int j = 0; j < gradient[0].length; j++) {
                gradient[i][j] *= gradientActivate[i][j];
            }
        }
        double[][] dInput = Arithmetic.dot(gradient, Arithmetic.transpose(weight));
        double[][] dWeight = Arithmetic.dot(Arithmetic.transpose(lastLayer), gradient);
        double[] dBias = new double[sizeOut];
        for (int j = 0; j < sizeOut; j++) {
            double sum = 0;
            for (int i = 0; i < gradient.length; i++) {
                sum += gradient[i][j];
            }
            dBias[j] = sum;
        }
        optimizer.update(weight,bias,dWeight,dBias,learningRate);
        return dInput;
    }

    public double[][] getWeight() {
        return weight;
    }
    public double[] getBias() {
        return bias;
    }
    public int getSizeOut(){
        return sizeOut;
    }
}
