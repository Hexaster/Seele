package Seele.Optimizer;

import Seele.Layer;

public interface Optimizer {
    void update(double[][]weight, double[] bias, double[][] dWeight, double[] dBias, double learningRate);
}
