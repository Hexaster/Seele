package Seele.LossFunction;

public class CrossEntropy implements LossFunction{
    @Override
    public double getLoss(double[][] output, double[][] target) {
        double result = 0;
        int totalElements = 0;
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                result += target[i][j] * Math.log(output[i][j]);
                totalElements++;
            }
        }
        return -result / totalElements;
    }

    @Override
    public double[][] gradient(double[][] output, double[][] target) {
        double[][] lossPrime = new double[output.length][output[0].length];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[0].length; j++) {
                lossPrime[i][j] = -(target[i][j] / output[i][j]);
            }
        }
        return lossPrime;
    }
}
