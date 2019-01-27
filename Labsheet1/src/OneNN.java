
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
 * @author mattp
 */
public class OneNN extends AbstractClassifier{
    public Instances data = null;
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        this.data = i;
    }
    
    public void classifyInstance(Instances i){
        for(Instance x: this.data){
            
        }
    }
    
    public double distance(Instance a, Instance b){
        a.
    }
    
}
