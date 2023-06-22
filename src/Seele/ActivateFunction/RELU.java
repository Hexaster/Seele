package Seele.ActivateFunction;

/**
 * The function:
 * f(x) = a*x  x>=0
 *        b*x  x<0
 * The default values of a and b are 1 and 0 respectively, so it is the normal RELU.
 * You can change their value freely, for example, change the value of b to
 * get Leaky RELU.
 */
public class RELU implements ActivateFunction{
    public double a;
    public double b;
    public RELU(){
        this.a = 1;
        this.b = 0;
    }
    public RELU(double a, double b){
        this.a = a;
        this.b = b;
    }

    @Override
    public double[][] actForward(double[][] x) {
        int m = x.length;
        int n = x[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = x[i][j] > 0 ? a * x[i][j] : b * x[i][j];
            }
        }
        return result;
    }

    @Override
    public double[][] actBackward(double[][] x) {
        int m = x.length;
        int n = x[0].length;
        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = x[i][j] > 0 ? a : b;
            }
        }
        return result;
    }
}
