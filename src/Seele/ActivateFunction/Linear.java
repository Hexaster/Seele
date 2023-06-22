package Seele.ActivateFunction;

public class Linear implements ActivateFunction {

    @Override
    public double[][] actForward(double[][] x) {
        return x;
    }

    @Override

    public double[][] actBackward(double[][] x) {
        double[][] result = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                result[i][j] = 1;
            }
        }
        return result;
    }


    /*
    public double actBackward(double x){
        return 1;
    }

     */
}
