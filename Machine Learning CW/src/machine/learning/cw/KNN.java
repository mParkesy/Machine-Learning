/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine.learning.cw;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author xze15agu
 */
public class KNN extends AbstractClassifier {
    private Instances data = null;
    private int k = 1;
    private boolean flag = true;
    private boolean leave = false;
    private boolean voting = false;
    
    public KNN() {

    }
    
    public KNN(boolean flag, boolean leave, boolean voting){
        this.flag = flag;
        this.leave = leave;
        this.voting = voting;
    }

    public Instances getData() {
        return data;
    }

    public int getK() {
        return k;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    public void setK(int k) {
        this.k = k;
    }

    public void setFlag(boolean flag) {
        this.flag = flag;
    }

    public void setLeave(boolean leave) {
        this.leave = leave;
    }
    
    public void setVoting(boolean vote) {
        this.voting = vote;
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        this.data = i;
        
        if(leave){
            // k needs to be set
            int initialK = (int) Math.rint(i.numInstances() * 0.2);
            int maxK = Math.min(100, initialK);
            Instances train = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TRAIN.arff");
            Instances test = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TEST.arff");
            train.setClassIndex(4);
            test.setClassIndex(4);
            
            double[] accuracies = new double[maxK+1];
            for(int n = 1; n <= maxK; n++){
                KNN classifier = new KNN();
                classifier.setK(n);
                System.out.println("K: " + classifier.getK());
                classifier.setLeave(false);
                classifier.buildClassifier(train);
                accuracies[n] = getAccuracey(test, classifier);
            }
            this.k = maxPosInDoubleArray(accuracies);
        }
        
        if(flag) {
            standardiseAttributes(i);
//            double total = 0.0;
//            for(int n = 0; n < this.data.numAttributes(); n++){
//                if(i.classIndex() != n){
//                    for(Instance x : this.data){
//                        total += x.value(n);
//                    }
//                    double mean = total/this.data.numInstances();
//                    double sdTotal = 0.0;
//                    for(Instance x : this.data){
//                        sdTotal += Math.pow(x.value(n) - mean, 2);
//                        //x.setValue(n, SampleDeviation(x, mean, this.data.numInstances()));
//                    }   
//                    double sd = Math.sqrt(sdTotal / this.data.numInstances());
//                    for(Instance x : this.data){
//                        x.setValue(n, ((x.value(n) - mean)/ sd));
//                    }
//                }
//            }
        }
    }
    
    public void standardiseAttributes(Instances i){
        double total = 0.0;
        for(int n = 0; n < this.data.numAttributes(); n++){
            if(i.classIndex() != n){
                for(Instance x : this.data){
                    total += x.value(n);
                }
                double mean = total/this.data.numInstances();
                double sdTotal = 0.0;
                for(Instance x : this.data){
                    sdTotal += Math.pow(x.value(n) - mean, 2);
                    //x.setValue(n, SampleDeviation(x, mean, this.data.numInstances()));
                }   
                double sd = Math.sqrt(sdTotal / this.data.numInstances());
                for(Instance x : this.data){
                    x.setValue(n, ((x.value(n) - mean)/ sd));
                }
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance i){
        return maxPosInDoubleArray(standardVote(i));
    }
    
        @Override
    public double[] distributionForInstance(Instance i){
        return standardVote(i);
    }
    
    public double[] standardVote(Instance i){
        TreeMap<Double, List<Instance>> map = new TreeMap<>(Collections.reverseOrder());
        int neighbourCount = 0;
        // loop over all instances
        for(Instance x: this.data){
            // get distance between instance to be classified and current in list
            double dis = distance(i, x);
            // define a new list
            List<Instance> list = new ArrayList<>();
            // check to see whether neighbour count is equal to k
            if(neighbourCount < this.k){
                // check to see if that distance has already been calculated
                if(map.containsKey(dis)){
                    // add the current 
                   map.get(dis).add(x);
                   neighbourCount++;
                } else {
                   list.add(x);
                   neighbourCount++;
                   map.put(dis, list);
                }
            } else {
                if(dis < map.lastKey()){
                    if(map.containsKey(dis)){
                        map.get(dis).add(x);
                        neighbourCount++;
                    } else {
                        map.pollLastEntry();
                        list.add(x);
                        map.put(dis, list);
                    }
                }
            }
        }
        
        double[] voting = new double[this.data.numClasses()];
        for(Map.Entry<Double, List<Instance>> entry : map.entrySet()) {
            List<Instance> value = entry.getValue();
            for(Instance y : value){
                voting[(int)y.classValue()]++;
            }
            
            //System.out.println(key + " => " + value.toString());
            //System.out.println("-----");
            double total = 0.0;
            
            for(double num : voting){
                total += num;
            }
            for(int x = 0; x < voting.length; x++){
                voting[x] = voting[x] / total;
            }
        }
        return voting;
    }

    public int maxPosInDoubleArray(double[] list){
        int maxPos = 0;
        for(int i = 0; i < list.length; i++){
            if ( list[i] > list[maxPos] ) {
                maxPos = i;
            } else if (list[i] == list[maxPos]){
                Random random = new Random(System.currentTimeMillis());
                if (random.nextBoolean()) {
                    maxPos = i;
                }
            }
        }
        return maxPos;
    }
    
    public double distance(Instance a, Instance b){
        double[] list1 = a.toDoubleArray();
        double[] list2 = b.toDoubleArray();
        double sum = 0;
        for (int i = 0; i < list1.length; i++){
            if(i != a.classIndex()){
                sum += Math.pow(list1[i] - list2[i], 2);
            }
        }
        return Math.sqrt(sum);
    }
    
    public static Instances[] splitData(Instances all, double proportion){
        Random r = new Random(System.currentTimeMillis());
        all.randomize(r);
        Instances[] split = new Instances[2];
        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        int n = (int) Math.round(proportion * all.numInstances());
        int b = split[0].numInstances();
        
        while(split[0].numInstances() != n){
            split[1].add(split[0].get(n));
            split[0].remove(n);
        }        
        return split;
    }

    public static double getAccuracey(Instances test, KNN classifier){
        double count = 0;
        double correct = 0;
        
        for(Instance x: test){
            double result = classifier.classifyInstance(x);
            //System.out.println(Arrays.toString(classifier.distributionForInstance(x)));
            //double[] dis = classifier.distributionForInstance(x);
//            for(int i = 0; i < dis.length; i++){
//                System.out.println(dis[i]);
//            }
            
            double actual = x.classValue();
            if(result == actual){
                correct++;
            }            
            count++;
            //System.out.println("Distribtuin: " + Arrays.toString(classifier.distributionForInstance(x)));
        }
        double acc = ((correct/count)*100);
        
        return acc;        
    }
    
    public static void main(String[] args) throws Exception {
        
        //Instances train = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TRAIN.arff");
        //Instances test = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood_TEST.arff");
        //Instances all = MachineLearningCW.loadData("U:/Documents/NetBeansProjects/Machine Learning CW/blood/blood.arff");
        Instances all = MachineLearningCW.loadData("E:/Documents/NetBeansProjects/Machine Learning/Machine Learning CW/blood/blood.arff");
        //train.setClassIndex(4);
        //test.setClassIndex(4);
        
        Instances[] splitAll = splitData(all, 0.7);
        Instances all_train = splitAll[0];
        Instances all_test = splitAll[1];
        
        all.setClassIndex(4);
        KNN classifier = new KNN();
        classifier.setK(1);
        
        all_train.setClassIndex(all_train.numAttributes()-1);
        all_test.setClassIndex(all_train.numAttributes()-1);
        
        //classifier.buildClassifier(train);
        classifier.setLeave(false);
        classifier.setFlag(true);
        classifier.buildClassifier(all_train);
        
        System.out.println(getAccuracey(all_test, classifier));
        
        
//        double count = 0;
//        double correct = 0;
//        for(Instance x: all_test){
//            double result = classifier.classifyInstance(x);
//            //double[] dis = classifier.distributionForInstance(x);
////            for(int i = 0; i < dis.length; i++){
////                System.out.println(dis[i]);
////            }
//            
//            double actual = x.classValue();
//            if(result == actual){
//                correct++;
//            }            
//            count++;
//            //System.out.println("Distribtuin: " + Arrays.toString(classifier.distributionForInstance(x)));
//        }
//        System.out.println("Accuracey: " + (correct/count)*100);
        
        
    }    
}
