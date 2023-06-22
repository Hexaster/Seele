package Seele;

/**
 * Some mathematical operations
 */
public class Arithmetic {
    public static double[][] dot(double[][] matrixA, double[][] matrixB) {
        int rowsA = matrixA.length;
        int colsA = matrixA[0].length;
        int rowsB = matrixB.length;
        int colsB = matrixB[0].length;
        if (colsA != rowsB) {
            throw new IllegalArgumentException("The number of columns in matrix A must be equal to the number of rows in matrix B");
        }
        double[][] result = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposedMatrix = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }

    public static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static double[] Mean(double[][] input){
        double[] mean = new double[input[0].length];
        for (int j = 0; j < input[0].length; j++) {
            for (int i = 0; i < input.length; i++) {
                mean[j] += input[i][j];
            }
            mean[j] /= input.length;
        }
        return mean;
    }

    public static double[] StandardDev(double[][] input){
        double[] mean = Mean(input);
        double[] standardDev = new double[input[0].length];
        for (int j = 0; j < input[0].length; j++) {
            for (int i = 0; i < input.length; i++) {
                standardDev[j] += Math.pow(input[i][j] - mean[j], 2);
            }
            standardDev[j] = Math.sqrt(standardDev[j] / (input.length - 1));
        }
        return standardDev;
    }

    public static double[][] normalisation(double[][] input){
        double[][] result = new double[input.length][input[0].length];
        double[] mean = Mean(input);
        double[] standardDev = StandardDev(input);
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                result[i][j] = (input[i][j] - mean[j]) / standardDev[j];
            }
        }
        return result;
    }
}
