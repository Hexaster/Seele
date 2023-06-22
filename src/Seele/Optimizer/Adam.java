package Seele.Optimizer;

public class Adam implements Optimizer{
    double epsilon = 0.0000000001;
    double beta1 = 0.1;
    double beta2 = 0.001;
    int t = 0;
    double[][] momentum;
    double[][] rms;
    double[] momentumBias;
    double[] rmsBias;

    public Adam(){
    }
    public Adam(double epsilon, double beta1, double beta2) {
        this.epsilon = epsilon;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public void update(double[][] weight, double[] bias, double[][] dWeight, double[] dBias, double learningRate) {
        if (momentum == null || rms == null) {
            momentum = new double[weight.length][weight[0].length];
            rms = new double[weight.length][weight[0].length];
        }
        if (momentumBias == null || rmsBias == null) {
            momentumBias = new double[bias.length];
            rmsBias = new double[bias.length];
        }
        t++;

        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[0].length; j++) {
                momentum[i][j] = (1 - beta1) * momentum[i][j] + beta1 * dWeight[i][j];
                rms[i][j] =  (1 - beta2) * rms[i][j] + beta2 * Math.pow(dWeight[i][j], 2);

                double momentumHat = momentum[i][j] / (1 - Math.pow(beta1, t));
                double rmsHat = rms[i][j] / (1 - Math.pow(beta2, t));

                weight[i][j] -= learningRate * momentumHat / (Math.sqrt(rmsHat) + epsilon);
            }
        }

        for (int j = 0; j < bias.length; j++) {
            momentumBias[j] = (1 - beta1) * momentumBias[j] + beta1 * dBias[j];
            rmsBias[j] =  (1 - beta2) * rmsBias[j] + beta2 * Math.pow(dBias[j], 2);
            double momentumHat = momentumBias[j] / (1 - Math.pow(beta1, t));
            double rmsHat = rmsBias[j] / (1 - Math.pow(beta2, t));
            bias[j] -= learningRate * momentumHat / (Math.sqrt(rmsHat) + epsilon);
        }
    }
}
