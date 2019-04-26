/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machine.learning.cw;

import java.io.FileReader;
import weka.core.Instances;

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
    public static void main(String[] args) {
        // TODO code application logic here
    }
    
}
