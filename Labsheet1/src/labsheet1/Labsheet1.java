/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package labsheet1;

import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author 100116544
 */
public class Labsheet1 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //Instances train = loadData("C:/Users/mattp/Documents/NetBeansProjects/Machine-Learning/Labsheet1/Arsenal_TRAIN.arff");
        //Instances test = loadData("C:/Users/mattp/Documents/NetBeansProjects/Machine-Learning/Labsheet1/Arsenal_TEST.arff");
        Instances train = loadData("E:/Documents/NetBeansProjects/Machine Learning/Labsheet1/Arsenal_TRAIN.arff");
        Instances test = loadData("E:/Documents/NetBeansProjects/Machine Learning/Labsheet1/Arsenal_TEST.arff");
        System.out.println("Number of instances in training data: " + train.numInstances());
        System.out.println("Number attributes in test data: " + test.numAttributes());
        
        System.out.println("Number of wins in training data: " + trainingWins(train));
        System.out.println("5th test data as array:");
        for(double d: fifthTestData(test)){
            System.out.print(d + ", ");
        }
        System.out.println("");
        
        System.out.println("Training instances toString: ");
        System.out.println(toString(train));
        
        System.out.println("");
        
        deleteAttribute(train, 2);
        deleteAttribute(test, 2);
        System.out.println("Attribute 2 deleted from training and test instances.");
        
        System.out.println("Train and test instances toString: ");
        System.out.println(toString(train));
        System.out.println("");
        System.out.println(toString(test));
        
        //train = loadData("C:/Users/mattp/Documents/NetBeansProjects/Machine-Learning/Labsheet1/Arsenal_TRAIN.arff");
        //test = loadData("C:/Users/mattp/Documents/NetBeansProjects/Machine-Learning/Labsheet1/Arsenal_TEST.arff");
        train = loadData("E:/Documents/NetBeansProjects/Machine Learning/Labsheet1/Arsenal_TRAIN.arff");
        test = loadData("E:/Documents/NetBeansProjects/Machine Learning/Labsheet1/Arsenal_TEST.arff");
        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(train.numAttributes()-1);
        System.out.println("Class: " + train.numClasses());
        
        
        System.out.println("Building classifiers...");
        Classifier bayes = buildNaive(train);
        Classifier neighbour = buildNeighbour(train);

        System.out.println("Naive Bayes Classifier accuracey is: "+ classifierAccuracey(test, bayes) +"%");
        System.out.println("Nearest Neighbour Classifier accuracey is: "+ classifierAccuracey(test, neighbour) +"%");
        
        System.out.println("Classifier Distribution: ");
        getDistribtuion(bayes, neighbour, test);
    }
    
    /**
     * Load instances objects from data file
     * @param path The path for the data file
     * @return The instances object
     */
    public static Instances loadData(String path){
        Instances train;
        try{
            FileReader reader = new FileReader(path);
            train = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
            train = null;
        } 
        return train;
    }
    
    /**
     * Sums up the number of wins in the training data
     * @param train The training data instance
     * @return An integer corresponding to the number of wins
     */
    public static int trainingWins(Instances train){
        int counter = 0;
        train.setClassIndex(3);
        for(Instance i: train){
            if(i.classValue() == 2){
                counter++;
            }
        }    
        return counter;
    }
    
    /**
     * Gets the fifth piece of instance in the test data
     * @param test The test data instances
     * @return The fifth instance returned as a double array
     */
    public static double[] fifthTestData(Instances test){
        return test.get(4).toDoubleArray();
    }
    
    /**
     * A to string method for a object of instances
     * @param data The instances object
     * @return The instances as a string
     */
    public static String toString(Instances data){
        return data.toString();
    }
    
    
    public static Instances deleteAttribute(Instances data, int x){
        data.deleteAttributeAt(x);
        return data;
    }
    
    public static Classifier buildNaive(Instances data){
        Classifier bayes = new NaiveBayes();
        try {
            bayes.buildClassifier(data);
        } catch (Exception ex) {
            System.out.println(ex);
        }  
        return bayes;
    }
    
    public static Classifier buildNeighbour(Instances data){
        Classifier neigh = new IBk();
        try {
            neigh.buildClassifier(data);
        } catch (Exception ex) {
            System.out.println(ex);
        }  
        return neigh;
    }
    
    public static double classifierAccuracey(Instances test, Classifier c){
        int counter = 0;
        for(Instance i: test){
            try {
                double value = c.classifyInstance(i);
                System.out.println("Predicted: " + value + " , Actual: " + i.classValue());
                if(i.classValue() == value){
                    counter++;
                }
            } catch (Exception ex) {
                System.out.println(ex);
            }
        }
        return ((double)counter/(double)test.numInstances())*100;
    }
    
    public static void getDistribtuion(Classifier naive, Classifier neigh, Instances data){
        for(Instance i: data){
            System.out.println("");
            try {
                printDoubleArray(naive.distributionForInstance(i));
                System.out.print("     ");
                printDoubleArray(neigh.distributionForInstance(i));
                System.out.print("     ");
            } catch (Exception ex) {
                System.out.println(ex);
            }
        }
    }
    
    public static void printDoubleArray(double[] data){
        DecimalFormat df = new DecimalFormat("0.00");
        for(double d: data){
            System.out.print(df.format(d)+ ",");
        }
    } 
}
