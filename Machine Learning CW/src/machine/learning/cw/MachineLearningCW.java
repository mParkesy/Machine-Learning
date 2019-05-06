/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine.learning.cw;

import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.List;
import static machine.learning.cw.KNN.splitData;
import static machine.learning.cw.KnnEnsemble.getFileNames;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.pmml.jaxbbindings.DecisionTree;

/**
 *
 * @author xze15agu
 */
public class MachineLearningCW {
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
     * A to string method for a object of instances
     * @param data The instances object
     * @return The instances as a string
     */
    public static String toString(Instances data){
        return data.toString();
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        int mode = 5;
        
        final File folder = new File("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/datasets/");
        List<String> fileList = getFileNames(folder);
        
        switch (mode) {
        // 1NN vs improvements
            case 1:
                break;
            case 2:
                //Instances train = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TRAIN.arff");
                //Instances test = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TEST.arff");
                //Instances all = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood.arff");
                //Instances all = MachineLearningCW.loadData("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/blood/blood.arff");
                //Instances all = MachineLearningCW.loadData("C:/Users/Parkesy/Documents/NetBeansProjects/Machine-Learning/Machine Learning CW/blood/blood.arff");
                //Instances train = MachineLearningCW.loadData("C:/Users/Parkesy/Documents/NetBeansProjects/Machine-Learning/Machine Learning CW/blood/blood_TRAIN.arff");
                //Instances test = MachineLearningCW.loadData("C:/Users/Parkesy/Documents/NetBeansProjects/Machine-Learning/Machine Learning CW/blood/blood_TEST.arff");
                //train.setClassIndex(4);
                //test.setClassIndex(4);
                

                //        try (PrintWriter writer = new PrintWriter(new File("1-NN Improvements.csv"))){
                //            StringBuilder sb = new StringBuilder();
                //            sb.append("dataset,");
                //            sb.append("leave,");
                //            sb.append("standardisation,");
                //            sb.append("weighting,");
                //            sb.append("accuracey,");
                //            sb.append("balancedAcc,");
                //            sb.append("f1,");
                //            sb.append("mcc,");
                //            sb.append("meanAUROC,");
                //            sb.append("n11,");
                //            sb.append("precision,");
                //            sb.append("recall,");
                //            sb.append("sensitivity,");
                //            sb.append("specificity,");
                //            sb.append("stddev\n");
                //            writer.write(sb.toString());
                //        } catch (FileNotFoundException e) {
                //            System.out.println(e.getMessage());
                //        }
                
                boolean[][] arr = new boolean[][]{
                    {false, false, false},
                    //{true, false, false},
                    //{false, true, false},
                    //{false, false, true},
                    //{true, true, true},
                    //{false, true, true},
                    //{true, true, false},
                    //{true, false, true},
                };  for(int y = 0; y < arr.length; y++){
                    
                    for(String path : fileList){
                        String filename = path.substring(path.lastIndexOf("\\")+1, path.indexOf("."));
                        Instances all = MachineLearningCW.loadData(path);
                        Instances split[] = splitData(all, 0.7);
                        Instances train = split[0];
                        Instances test = split[1];
                        
                        train.setClassIndex(train.numAttributes()-1);
                        test.setClassIndex(test.numAttributes()-1);
                        
                        KNN classifier = new KNN();
                        classifier.setK(1);
                        
                        classifier.setLeave(arr[y][0]);
                        classifier.setStandardisation(arr[y][1]);
                        classifier.setWeighting(arr[y][2]);
                        classifier.buildClassifier(train);
                        
                        SingleTestSetEvaluator st = new SingleTestSetEvaluator();
                        ClassifierResults res = st.evaluate(classifier, test);
                        double[] averages = new double[11];
                        averages[0] = res.getAcc();
                        averages[1] = res.balancedAcc;
                        averages[2] = res.f1;
                        averages[3] = res.mcc;
                        averages[4] = res.meanAUROC;
                        averages[5] = res.nll;
                        averages[6] = res.precision;
                        averages[7] = res.recall;
                        averages[8] = res.sensitivity;
                        averages[9] = res.specificity;
                        averages[10] = res.stddev;
                        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("1-NN Improvements.csv", true)))) {
                            StringBuilder sb = new StringBuilder();
                            sb.append(filename);
                            sb.append(",");
                            sb.append(arr[y][0]);
                            sb.append(",");
                            sb.append(arr[y][1]);
                            sb.append(",");
                            sb.append(arr[y][2]);
                            sb.append(",");
                            for(double x : averages){
                                sb.append(x);
                                sb.append(",");
                            }      
                            sb.append("\n");
                            System.out.println(sb.toString());
                            writer.write(sb.toString());
                        } catch (FileNotFoundException e) {
                            System.out.println(e.getMessage());
                        }
                    }
                }   
                break;
            // ensemble vs 1NN
            case 3:
                try (PrintWriter writer = new PrintWriter(new File("Ensemble vs 1NN.csv"))){
                    StringBuilder sb = new StringBuilder();
                    sb.append("dataset,");
                    sb.append("ensemble or 1nn,");
                    sb.append("accuracey,");
                    sb.append("balancedAcc,");
                    sb.append("f1,");
                    sb.append("mcc,");
                    sb.append("meanAUROC,");
                    sb.append("n11,");
                    sb.append("precision,");
                    sb.append("recall,");
                    sb.append("sensitivity,");
                    sb.append("specificity,");
                    sb.append("stddev\n");    
                    writer.write(sb.toString());  
                } catch (FileNotFoundException e) {
                    System.out.println(e.getMessage());
                }


                for(String path : fileList){
                    Instances all = MachineLearningCW.loadData(path);
                    KnnEnsemble ensemble = new KnnEnsemble();
                    Instances split[] = splitData(all, 0.7);
                    Instances train = split[0];
                    Instances test = split[1];
                    train.setClassIndex(train.numAttributes()-1);
                    test.setClassIndex(test.numAttributes()-1);
                    ensemble.buildEnsemble(train);
                    String filename = path.substring(path.lastIndexOf("\\")+1, path.indexOf(".")); 
                    ensemble.runEnsemble(test, filename);  
                }
                
                for(String path : fileList){
                    String filename = path.substring(path.lastIndexOf("\\")+1, path.indexOf("."));
                    Instances all = MachineLearningCW.loadData(path);
                    Instances split[] = splitData(all, 0.7);
                    Instances train = split[0];
                    Instances test = split[1];
                    train.setClassIndex(train.numAttributes()-1);
                    test.setClassIndex(test.numAttributes()-1);
                    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("Ensemble vs 1NN.csv", true)))) {
                        KNN classifier = new KNN();
                        classifier.setK(1);

                        classifier.setLeave(false);
                        classifier.setStandardisation(false);
                        classifier.setWeighting(false);
                        classifier.buildClassifier(train);
                        
                        SingleTestSetEvaluator st = new SingleTestSetEvaluator();
                        ClassifierResults res = st.evaluate(classifier, test);
                        double[] averages = new double[11];  
                        averages[0] = res.getAcc();
                        averages[1] = res.balancedAcc;
                        averages[2] = res.f1;
                        averages[3] = res.mcc;
                        averages[4] = res.meanAUROC;
                        averages[5] = res.nll;
                        averages[6] = res.precision;
                        averages[7] = res.recall;
                        averages[8] = res.sensitivity;
                        averages[9] = res.specificity;
                        averages[10] = res.stddev;
                        
                        StringBuilder sb = new StringBuilder();
                        sb.append(filename);
                        sb.append(",");
                        sb.append("1nn");
                        sb.append(",");
                        for(double x : averages){ 
                           sb.append(x);
                           sb.append(",");      
                        }
                        sb.append("\n");
                        System.out.println(sb.toString());    
                        writer.write(sb.toString());
                    } catch (FileNotFoundException e) {
                        System.out.println(e.getMessage());
                    }
                
                }
                break;
            // ensemble vs weka builds  
            case 4:
                try (PrintWriter writer = new PrintWriter(new File("Ensemble vs Weka.csv"))){
                    StringBuilder sb = new StringBuilder();
                    sb.append("dataset,");
                    sb.append("ensebmle or weka,");
                    sb.append("accuracey,");
                    sb.append("balancedAcc,");
                    sb.append("f1,");
                    sb.append("mcc,");
                    sb.append("meanAUROC,");
                    sb.append("n11,");
                    sb.append("precision,");
                    sb.append("recall,");
                    sb.append("sensitivity,");
                    sb.append("specificity,");
                    sb.append("stddev\n");    
                    writer.write(sb.toString());  
                } catch (FileNotFoundException e) {
                    System.out.println(e.getMessage());
                }

                for(String path : fileList){
                    Instances all = MachineLearningCW.loadData(path);
                    KnnEnsemble ensemble = new KnnEnsemble();
                    Instances split[] = splitData(all, 0.7);
                    Instances train = split[0];
                    Instances test = split[1];
                    train.setClassIndex(train.numAttributes()-1);
                    test.setClassIndex(test.numAttributes()-1);
                    ensemble.buildEnsemble(train);
                    String filename = path.substring(path.lastIndexOf("\\")+1, path.indexOf(".")); 
                    ensemble.runEnsemble(test, filename);  
                }
                
                for(String path : fileList){
                    String filename = path.substring(path.lastIndexOf("\\")+1, path.indexOf("."));
                    Instances all = MachineLearningCW.loadData(path);
                    Instances split[] = splitData(all, 0.7);
                    Instances train = split[0];
                    Instances test = split[1];
                    train.setClassIndex(train.numAttributes()-1);
                    test.setClassIndex(test.numAttributes()-1);
                    
                    SingleTestSetEvaluator st = new SingleTestSetEvaluator();
                    
                    RandomForest rf = new RandomForest();
                    rf.buildClassifier(train);
                    ClassifierResults forestResults = st.evaluate(rf, test);
                    
                    AdaBoostM1 ab = new AdaBoostM1();
                    ab.buildClassifier(train);
                    ClassifierResults abResults = st.evaluate(ab, test);
                    
                    J48 j = new J48();
                    j.buildClassifier(train);
                    ClassifierResults jResults = st.evaluate(j, test);
                    
                    NaiveBayes nb = new NaiveBayes();
                    nb.buildClassifier(train);
                    ClassifierResults nbResults = st.evaluate(nb, test);
                    
                    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("Ensemble vs Weka.csv", true)))) {                                           
                        StringBuilder sb = new StringBuilder();
                        sb.append(filename);
                        sb.append(",");
                        sb.append("random forest,");
                        sb.append(forestResults.getAcc());
                        sb.append(",");
                        sb.append(forestResults.balancedAcc);
                        sb.append(",");
                        sb.append(forestResults.f1);
                        sb.append(",");
                        sb.append(forestResults.mcc);
                        sb.append(",");
                        sb.append(forestResults.meanAUROC);
                        sb.append(",");
                        sb.append(forestResults.nll);
                        sb.append(",");
                        sb.append(forestResults.precision);
                        sb.append(",");
                        sb.append(forestResults.recall);
                        sb.append(",");
                        sb.append(forestResults.sensitivity);
                        sb.append(",");
                        sb.append(forestResults.specificity);
                        sb.append(",");
                        sb.append(forestResults.stddev);
                        sb.append("\n"); 
                        
                        sb.append(filename);
                        sb.append(",");
                        sb.append("ADA Boost,");
                        sb.append(abResults.getAcc());
                        sb.append(",");
                        sb.append(abResults.balancedAcc);
                        sb.append(",");
                        sb.append(abResults.f1);
                        sb.append(",");
                        sb.append(abResults.mcc);
                        sb.append(",");
                        sb.append(abResults.meanAUROC);
                        sb.append(",");
                        sb.append(abResults.nll);
                        sb.append(",");
                        sb.append(abResults.precision);
                        sb.append(",");
                        sb.append(abResults.recall);
                        sb.append(",");
                        sb.append(abResults.sensitivity);
                        sb.append(",");
                        sb.append(abResults.specificity);
                        sb.append(",");
                        sb.append(abResults.stddev);
                        sb.append("\n"); 
                        
                        sb.append(filename);
                        sb.append(",");
                        sb.append("J48,");
                        sb.append(jResults.getAcc());
                        sb.append(",");
                        sb.append(jResults.balancedAcc);
                        sb.append(",");
                        sb.append(jResults.f1);
                        sb.append(",");
                        sb.append(jResults.mcc);
                        sb.append(",");
                        sb.append(jResults.meanAUROC);
                        sb.append(",");
                        sb.append(jResults.nll);
                        sb.append(",");
                        sb.append(jResults.precision);
                        sb.append(",");
                        sb.append(jResults.recall);
                        sb.append(",");
                        sb.append(jResults.sensitivity);
                        sb.append(",");
                        sb.append(jResults.specificity);
                        sb.append(",");
                        sb.append(jResults.stddev);
                        sb.append("\n"); 
                        
                        sb.append(filename);
                        sb.append(",");
                        sb.append("Naive Bayes,");
                        sb.append(nbResults.getAcc());
                        sb.append(",");
                        sb.append(nbResults.balancedAcc);
                        sb.append(",");
                        sb.append(nbResults.f1);
                        sb.append(",");
                        sb.append(nbResults.mcc);
                        sb.append(",");
                        sb.append(nbResults.meanAUROC);
                        sb.append(",");
                        sb.append(nbResults.nll);
                        sb.append(",");
                        sb.append(nbResults.precision);
                        sb.append(",");
                        sb.append(nbResults.recall);
                        sb.append(",");
                        sb.append(nbResults.sensitivity);
                        sb.append(",");
                        sb.append(nbResults.specificity);
                        sb.append(",");
                        sb.append(nbResults.stddev);
                        sb.append("\n"); 
                        
                        writer.write(sb.toString());
                    } catch (FileNotFoundException e) {
                        System.out.println(e.getMessage());
                    }
                    
                    System.out.println(filename + " - forest: " + forestResults.getAcc());
                    System.out.println(filename + " - ab: " + abResults.getAcc());
                    System.out.println(filename + " - j: " + jResults.getAcc());
                    System.out.println("");
                }                    
                break;
            case 5:
                try (PrintWriter writer = new PrintWriter(new File("Blood Transfusion.csv"))){
                    StringBuilder sb = new StringBuilder();
                    sb.append("accuracy,");
                    sb.append("balancedAcc,");
                    sb.append("f1,");
                    sb.append("mcc,");
                    sb.append("meanAUROC,");
                    sb.append("n11,");
                    sb.append("precision,");
                    sb.append("recall,");
                    sb.append("sensitivity,");
                    sb.append("specificity,");
                    sb.append("stddev\n");    
                    writer.write(sb.toString());  
                } catch (FileNotFoundException e) {
                    System.out.println(e.getMessage());
                }
                
                    Instances all = MachineLearningCW.loadData("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/datasets/blood/blood.arff");
                    Instances split[] = splitData(all, 0.7);
                    Instances train = split[0];
                    Instances test = split[1];
                    train.setClassIndex(train.numAttributes()-1);
                    test.setClassIndex(test.numAttributes()-1);
                    
                    SingleTestSetEvaluator st = new SingleTestSetEvaluator();
                    //NaiveBayes rf = new NaiveBayes();
                    J48 rf = new J48();
                    
                    
                    rf.buildClassifier(train);
                    ClassifierResults forestResults = st.evaluate(rf, test);
                    System.out.println(forestResults.allPerformanceMetricsToString());
                    try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("Blood Transfusion.csv", true)))) {      
                        StringBuilder sb = new StringBuilder();
                        sb.append(forestResults.getAcc());
                        sb.append(",");
                        sb.append(forestResults.balancedAcc);
                        sb.append(",");
                        sb.append(forestResults.f1);
                        sb.append(",");
                        sb.append(forestResults.mcc);
                        sb.append(",");
                        sb.append(forestResults.meanAUROC);
                        sb.append(",");
                        sb.append(forestResults.nll);
                        sb.append(",");
                        sb.append(forestResults.precision);
                        sb.append(",");
                        sb.append(forestResults.recall);
                        sb.append(",");
                        sb.append(forestResults.sensitivity);
                        sb.append(",");
                        sb.append(forestResults.specificity);
                        sb.append(",");
                        sb.append(forestResults.stddev);
                        sb.append("\n"); 
                        writer.write(sb.toString()); 
                    } catch (FileNotFoundException e) {
                       System.out.println(e.getMessage());
                    }
                
                break;
            default:
                break;    
        }
    }
    
    
}
