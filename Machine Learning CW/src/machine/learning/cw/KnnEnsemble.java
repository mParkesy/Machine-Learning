/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine.learning.cw;

import java.util.ArrayList;
import java.util.Random;
import static machine.learning.cw.KNN.getAccuracey;
import static machine.learning.cw.KNN.splitData;
import weka.core.Instances;

/**
 *
 * @author mattp
 */
public class KnnEnsemble {
    private int size = 50;
    private KNN[] list = new KNN[size];
    
    public void startEnsemble() throws Exception{
        // load in all data
        Instances all = MachineLearningCW.loadData("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/blood/blood.arff");
        // loop over array of classifiers
        for(int i = 0; i < this.size; i++){
            // initialize a new KNN
            list[i] = new KNN();
            // split the data into training and test data
            Instances[] splitAll = splitData(all, myRandom(0.6, 0.8));
            Instances all_train = splitAll[0];
            Instances all_test = splitAll[1];
            
            // set the class index
            all.setClassIndex(4);
            all_train.setClassIndex(all_train.numAttributes()-1);
            all_test.setClassIndex(all_train.numAttributes()-1);
            
            // set flags
            list[i].setLeave(false);
            list[i].setFlag(false);
            list[i].setVoting(false);
            list[i].buildClassifier(all_train);
            
            // get accuracey of classifier
            System.out.println(getAccuracey(all_test, list[i]));
        }
    }
    
    /**
     * Gets a random double between two values
     * @param min The min value
     * @param max The max value
     * @return A value between min and max
     */
    double myRandom(double min, double max) {
        Random r = new Random();
        return (r.nextInt((int)((max-min)*10+1))+min*10) / 10.0;
    }
    
    public static void main(String[] args) throws Exception {
        KnnEnsemble test = new KnnEnsemble();
        test.startEnsemble();
        
    }
}


//Instances all = MachineLearningCW.loadData("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/blood/blood.arff");
////train.setClassIndex(4);
////test.setClassIndex(4);
//
//Instances[] splitAll = splitData(all, 0.7);
//Instances all_train = splitAll[0];
//Instances all_test = splitAll[1];
//
//all.setClassIndex(4);
//KNN classifier = new KNN();
//classifier.setK(1);
//
//all_train.setClassIndex(all_train.numAttributes()-1);
//all_test.setClassIndex(all_train.numAttributes()-1);
//
////classifier.buildClassifier(train);
//classifier.setLeave(false);
//classifier.setFlag(true);
//classifier.buildClassifier(all_train);
//
//System.out.println(getAccuracey(all_test, classifier));
        