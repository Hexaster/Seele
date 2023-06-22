package Seele.ActivateFunction;

public class Sigmoid implements ActivateFunction{
    private double[][] sigmoidMatrix(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = 1.0 / (1.0 + Math.exp(-x[i][j]));
            }
        }
        return result;
    }

    @Override
    public double[][] actForward(double[][] x) {
        return sigmoidMatrix(x);
    }


    @Override
    public double[][] actBackward(double[][] x) {
        double[][] sigmoid = sigmoidMatrix(x);
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = sigmoid[i][j] * (1 - sigmoid[i][j]);
            }
        }
        return result;
    }
}

