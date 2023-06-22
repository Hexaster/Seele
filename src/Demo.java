import Seele.ActivateFunction.*;
import Seele.Arithmetic;
import Seele.DataLoader;
import Seele.LossFunction.*;
import Seele.NeuralNetwork;
import Seele.Optimizer.SGD;

import java.util.Arrays;
import java.util.OptionalInt;

/**
 * Seele V 0.01
 * This is the Demo of Seele. Please note that Seele is just a simple project of a
 * student in which robustness and optimisation are ignored. Furthermore, I
 * only did very limit simple tests, so the reliance of such a model is not
 * guaranteed.
 * If you have good suggestions, please feel free to contact me through:
 * lagrangedoge3@gmail.com, give me your suggestion, teach me why I need to do that,
 * thank you.
 *
 * Andrew
 * 22/6/2023
 */
public class Demo {
    public static void main(String[] args) {
        //Very simple dataset
        double[][] input = {
                {2,1},
                {1,3},
                {0,-5},
                {-7,-3}
        };
        double[][] target = {
                {4},
                {7},
                {-10},
                {-13}
        };

        //Normalise the dataset
        //input = Arithmetic.normalisation(input);

        //Create dataloader, set the batch size as 1 and shuffle
        DataLoader dataLoader = new DataLoader(input,target,1,true);
        //You can reset the batch size as well
        dataLoader.setBatchSize(2);


        //Create a Neural network, set learning rate and loss function.
        NeuralNetwork nn = new NeuralNetwork(dataLoader,0.01, new MSELoss());
        //Add layers to the network
        //nn.addLayer(6, new RELU(1,0), new SGD());
        nn.addLayer(target[0].length, new Linear());
        nn.train(100, OptionalInt.of(10));
        //System.out.println(Arrays.deepToString(nn.layers.get(0).getWeight()));
        //System.out.println(Arrays.toString(nn.layers.get(0).getBias()));

        //After training finishes, let's predict something
        double[][] test = {{4,-3},{5,2}};
        double[][] a = nn.predict(test);
        Arithmetic.printMatrix(a);
    }
}