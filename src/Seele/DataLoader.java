package Seele;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

/**
 * A DataLoader object contains a map of indices and batches.
 * A batch stores a part of dataset's features and targets.
 */
public class DataLoader {
    private double[][] features;
    private double[][] targets;
    private int batchSize;
    private boolean shuffle;
    private ArrayList<Integer> indices;
    public HashMap<Integer, Batch> batches = new HashMap<>();

    public class Batch{
        private double[][] features;
        private double[][] targets;

        public Batch(double[][] features, double[][] targets) {
            this.features = features;
            this.targets = targets;
        }
        public double[][] getFeatures() {
            return this.features;
        }

        public double[][] getTargets() {
            return this.targets;
        }


    }

    /**
     * Create a DataLoader to do operations on dataset
     * @param batchSize, the size of a batch, in default, the whole dataset is a batch
     * @param shuffle, whether we shuffle the dataset. In default, the value is false
     */
    public DataLoader(double[][] features, double[][] targets, int batchSize, boolean shuffle) {
        this.features = features;
        this.targets = targets;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            indices.add(i);
        }
        if (shuffle) {
            Collections.shuffle(indices, new Random());
        }
        this.setBatchSize(batchSize);
    }
    public DataLoader(double[][] features, double[][] targets) {
        this.features = features;
        this.targets = targets;
        this.batchSize = features.length;
        this.shuffle = false;
        this.indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            indices.add(i);
        }
        this.setBatchSize(batchSize);
    }
    public DataLoader(double[][] features, double[][] targets, int batchSize) {
        this.features = features;
        this.targets = targets;
        this.batchSize = batchSize;
        this.shuffle = false;
        this.indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            indices.add(i);
        }
        this.setBatchSize(batchSize);
    }
    public DataLoader(double[][] features, double[][] targets, boolean shuffle) {
        this.features = features;
        this.targets = targets;
        this.shuffle = shuffle;
        this.batchSize = features.length;
        this.indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            indices.add(i);
        }
        if (shuffle) {
            Collections.shuffle(indices, new Random());
        }
        this.setBatchSize(batchSize);
    }
    public DataLoader(double[][] features){
        this.features = features;
    }

    public void setBatchSize(int newBatchSize){
        this.batchSize = newBatchSize;
        int numBatch = (int) Math.ceil((double) features.length / batchSize);
        this.setBatches(numBatch);
    }


    private Batch nextBatch(int i) {
        int start = i * batchSize;
        int end = Math.min(start + batchSize, features.length);
        int batch_size = end - start;

        double[][] batchFeatures = new double[batch_size][];
        double[][] batchTargets = new double[batch_size][];

        for (int j = start; j < end; j++) {
            int idx = indices.get(j);
            batchFeatures[j - start] = features[idx];
            batchTargets[j - start] = targets[idx];
        }
        return new Batch(batchFeatures, batchTargets);
    }

    private void setBatches(int numBatch){
        for (int i = 0; i < numBatch; i++){
            batches.put(i,nextBatch(i));
        }
    }

    public int getBatchSize(){
        return this.batchSize;
    }

    public double[][] getFeatures() {
        return features;
    }

    public double[][] getTargets() {
        return targets;
    }
}
