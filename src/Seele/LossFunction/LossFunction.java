package Seele.LossFunction;

public interface LossFunction {

    /**
     * This method is simply used to print loss
     */
    double getLoss(double[][] output, double[][] target);

    /**
     * Used for backward pass
     * @param output
     * @param target
     * @return
     */
    double[][] gradient(double[][] output, double[][] target);
}
