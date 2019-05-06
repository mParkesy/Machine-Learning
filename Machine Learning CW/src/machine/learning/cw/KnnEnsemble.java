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
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static machine.learning.cw.KNN.getAccuracey;
import static machine.learning.cw.KNN.splitData;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author mattp
 */
public class KnnEnsemble {
    private int size = 50;
    private KNN[] list = new KNN[size];
    private Instances trainData;
    private Random rand = new Random();
    
    
    public void buildEnsemble(Instances train) throws Exception{
        this.trainData = train;
        for(int i = 0; i < this.size; i++){
            list[i] = new KNN();
            Instances subset = new Instances(train, 0);
            for(int j = 0; j < this.trainData.size(); j++){
                Instance get = train.get(rand.nextInt((train.size()-1 - 0) + 1) + 0);
                subset.add(get);    
            }
            
            list[i].setLeave(false);
            list[i].setStandardisation(false);
            list[i].setWeighting(false);
            list[i].buildClassifier(subset);
        }
    }
    
    public void runEnsemble(Instances test, String filename) throws Exception{        
        double[] averages = new double[11];                
        for(int i = 0; i < this.size; i++){
            //System.out.println("NN " + i + " : " + avg);                
            SingleTestSetEvaluator st = new SingleTestSetEvaluator();
            ClassifierResults res = st.evaluate(list[i], test);
            
            averages[0] += res.getAcc();
            averages[1] += res.balancedAcc;
            averages[2] += res.f1;
            averages[3] += res.mcc;
            averages[4] += res.meanAUROC;
            averages[5] += res.nll;
            averages[6] += res.precision;
            averages[7] += res.recall;
            averages[8] += res.sensitivity;
            averages[9] += res.specificity;
            averages[10] += res.stddev;
        }
        
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter("Ensemble vs 1NN.csv", true)))) {
            StringBuilder sb = new StringBuilder();
            sb.append(filename);
            sb.append(",");
            sb.append("ensemble");
            sb.append(",");
            for(double x : averages){
               x = x / this.size; 
               sb.append(x);
               sb.append(",");      
            }
            sb.append("\n"); 
            writer.write(sb.toString());  
        } catch (FileNotFoundException e) {
            System.out.println(e.getMessage());
        }
    }
   
    
    /**
     * Gets a random double between two values
     * @param min The min value
     * @param max The max value
     * @return A value between min and max
     */
    double randomDouble(double min, double max) {
        Random r = new Random();
        return (r.nextInt((int)((max-min)*10+1))+min*10) / 10.0;
    }

public static List<String> getFileNames(final File folder) {
    List<String> fileList = new ArrayList<>();
    for (final File fileEntry : folder.listFiles()) {
        if (fileEntry.isDirectory()) {
            for(final File d : fileEntry.listFiles()){
                if((!d.getName().contains("_TRAIN")) && (!d.getName().contains("_TEST"))){
                  fileList.add(d.getPath());
                }
            }
        }
    }
    return fileList;
}
    
public static void main(String[] args) throws Exception {
    final File folder = new File("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/datasets/");
    List<String> fileList = getFileNames(folder);

    try (PrintWriter writer = new PrintWriter(new File("1-NN Improvements.csv"))){
        StringBuilder sb = new StringBuilder();
        sb.append("dataset,");
        sb.append("leave,");
        sb.append("standardisation,");
        sb.append("weighting,");
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
}
        
        
//        KnnEnsemble ensemble = new KnnEnsemble();
//        Instances train = MachineLearningCW.loadData("C:/Users/Parkesy/Documents/NetBeansProjects/Machine-Learning/Machine Learning CW/blood/blood_TRAIN.arff");
//        Instances test = MachineLearningCW.loadData("C:/Users/Parkesy/Documents/NetBeansProjects/Machine-Learning/Machine Learning CW/blood/blood_TEST.arff");
//        ensemble.buildEnsemble(train);
//        ensemble.runEnsemble(test);
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
//classifier.setStandardisation(true);
//classifier.buildClassifier(all_train);
//
//System.out.println(getAccuracey(all_test, classifier));
        