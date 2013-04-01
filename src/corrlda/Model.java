/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package corrlda;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

/**
 *
 * @author kaldr
 */
public class Model {
    //Dataset preprocessed

    public int visualVocabSize;
    public int numImages;
    public int annotVocabSize;
    public int kfold;
    public int numFolds;
    public int numTagsToRetrieve;
    public ArrayList<ArrayList<Integer>> visualWordsInImages;
    public ArrayList<ArrayList<Integer>> annotWordsInImages;
//    public int[][] trainingFolds;
//    public int[][] testingFolds;
    public ArrayList<ArrayList<Integer>> trainingFolds;
    public ArrayList<ArrayList<Integer>> testingFolds;
    
    private static final String dataDir = "./data/";

    public Model() {
    	visualWordsInImages = new ArrayList<ArrayList<Integer>>();
    	annotWordsInImages = new ArrayList<ArrayList<Integer>>();
        visualVocabSize = 0;
        numImages = 0;
        annotVocabSize = 0;
        kfold = 1;
        numFolds = 0;
        numTagsToRetrieve = 10;
    }
    
    public Model(int k) {
    	this();
    	kfold = k;
    }
    
    public Model(int k, int numTags) {
    	this(k);
    	numTagsToRetrieve = numTags;
    }
    
    public void getTestingSets(int idx, int level, ArrayList<Integer> listSoFar) {
    	if(level==0) {
    		testingFolds.add(listSoFar);
    		return;
    	}
    	for(int i=idx; i<numImages; ++i) {
    		ArrayList<Integer> tempList = new ArrayList<Integer>();
    		tempList.addAll(listSoFar);
    		tempList.add(i);
    		getTestingSets(idx+1, level-1, tempList);
    	}
    }
    
    public boolean inTestSet(int idx, int fold) {
    	ArrayList<Integer> testSet = testingFolds.get(fold);
    	for(int i=0; i<kfold; ++i) {
    		if(testSet.get(i)==idx)
    			return true;
    	}
    	return false;
    }
    
    public void populateFolds() {
    	trainingFolds = new ArrayList<ArrayList<Integer>>();
    	testingFolds = new ArrayList<ArrayList<Integer>>();
    	getTestingSets(0, kfold, new ArrayList<Integer>());
    	numFolds = testingFolds.size();
        for(int i=0; i<numFolds; ++i) {
        	ArrayList<Integer> trainSet = new ArrayList<Integer>();
        	for(int j=0; j<numImages; ++j) {
        		if(!inTestSet(j, i))
        			trainSet.add(j);        		
        	}
        	trainingFolds.add(trainSet);
        }
    }

    public void initialize() throws IOException {
    	System.out.println("Initializing model...");
    	String visualWordFile = "featureVector.dat";
        String annotWordFile = "annotVector.dat";
        readFeatureVector(visualWordFile);
        readAnnotVector(annotWordFile);
    	populateFolds();
        
        System.out.println("Dataset has " + visualVocabSize + " visual words, " + numImages + " images, and" + annotVocabSize + " tags.\n ");
        System.out.println("The model is initialized.");
    }
    
    public void readFeatureVector(String filename) throws IOException {
    	BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(dataDir+filename)));
    	String line;
    	while((line = reader.readLine()) != null) {
    		StringTokenizer tknr = new StringTokenizer(line, ",");
    		if(visualVocabSize==0)
    			visualVocabSize = tknr.countTokens();
    		ArrayList<Integer> featureVector = new ArrayList<Integer>();
    		for(int i=0; tknr.hasMoreTokens(); ++i) {
    			int numTokens = Integer.parseInt(tknr.nextToken());
    			if(numTokens > 0) {
    				for(int j=0; j<numTokens; ++j)
    					featureVector.add(i);
    			}
    		}
			visualWordsInImages.add(featureVector);
    	}
    	numImages = visualWordsInImages.size();
    }
    
    public void readAnnotVector(String filename) throws IOException {
    	BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(dataDir+filename)));
    	String line;
    	while((line = reader.readLine()) != null) {
    		StringTokenizer tknr = new StringTokenizer(line, "\t");
    		if(annotVocabSize==0)
    			annotVocabSize = tknr.countTokens();
    		ArrayList<Integer> annotVector = new ArrayList<Integer>();
    		for(int i=0; tknr.hasMoreTokens(); ++i) {
    			boolean tokenPresent =Integer.parseInt(tknr.nextToken())==1;
    			if(tokenPresent) {
    				annotVector.add(i);
    			}
    		}
			annotWordsInImages.add(annotVector);
    	}
    }
}
