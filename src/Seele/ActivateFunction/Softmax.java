package Seele.ActivateFunction;

public class Softmax implements ActivateFunction{
    @Override
    public double[][] actForward(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            double max = x[i][0];
            for (int j = 1; j < x[0].length; j++) {
                max = Math.max(max, x[i][j]);
            }

            double sum = 0;
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = Math.exp(x[i][j] - max);
                sum += result[i][j];
            }

            for (int j = 0; j < x[0].length; j++) {
                result[i][j] /= sum;
            }
        }
        return result;
    }

    @Override
    public double[][] actBackward(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        double[][] softmax = actForward(x);
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                for (int k = 0; k < x[0].length; k++) {
                    if (j == k) {
                        result[i][j] += softmax[i][j] * (1 - softmax[i][k]);
                    } else {
                        result[i][j] -= softmax[i][j] * softmax[i][k];
                    }
                }
            }
        }
        return result;
    }
}
