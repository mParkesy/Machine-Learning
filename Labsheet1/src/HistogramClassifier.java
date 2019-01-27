
import weka.classifiers.Classifier;
import weka.core.Capabilities;
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
public class HistogramClassifier implements Classifier{
    private int bins;
    private int position;
    
    public HistogramClassifier(){
        this.bins = 10;
        this.position = 0;
    }
    
    public HistogramClassifier(int b, int p){
        this.bins = b;
        this.position = p;
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        int y = i.numClasses();
        
         
    } 

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * @return the bins
     */
    public int getBins() {
        return bins;
    }

    /**
     * @param bins the bins to set
     */
    public void setBins(int bins) {
        this.bins = bins;
    }

    /**
     * @return the position
     */
    public int getPosition() {
        return position;
    }

    /**
     * @param position the position to set
     */
    public void setPosition(int position) {
        this.position = position;
    }
    
}
