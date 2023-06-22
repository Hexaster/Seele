package Seele.LossFunction;

public class MSELoss implements LossFunction{
    @Override
    public double getLoss(double[][] output, double[][] target) {
        double result = 0;
        int totalElements = 0;
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                double diff = output[i][j] - target[i][j];
                result += Math.pow(diff, 2);
                totalElements++;
            }
        }
        return result/totalElements;
    }

    @Override
    public double[][] gradient(double[][] output, double[][] target) {
        double[][] lossPrime = new double[output.length][output[0].length];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                lossPrime[i][j] = 2 * (output[i][j] - target[i][j]);
            }
        }
        return lossPrime;
    }
}

