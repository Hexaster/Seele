package Seele.Optimizer;


public class SGD implements Optimizer{
    @Override
    public void update(double[][]weight, double[] bias, double[][] dWeight, double[] dBias, double learningRate) {
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[0].length; j++) {
                weight[i][j] -= learningRate * dWeight[i][j];
            }
        }

        for (int j = 0; j < bias.length; j++) {
            bias[j] -= learningRate * dBias[j];
        }
    }
}
