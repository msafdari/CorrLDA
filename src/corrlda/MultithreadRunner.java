package corrlda;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

public class MultithreadRunner implements Runnable{
	
	private static final String dataDir = "./data/logLikelihood/";
	private static int threadCounter = 0;
	
	private double alpha;
	private double beta;
	private double gamma;
	private int nitter;
	private Model model;
	private int threadID;
	
	public MultithreadRunner(Model m) {
		model = m;
		alpha = 1.0;
		beta = 2.0;
		gamma = 1.0;
		nitter = 10;
		threadID = threadCounter++;
	}
	
	public MultithreadRunner(Model m, double a, double b, double g, int iter) {
		this(m);
		alpha = a;
		beta = b;
		gamma = g;
		nitter = iter;
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		System.out.println("Starting thread " + threadID);
		ArrayList<Double> logLik = new ArrayList<Double>(model.numFolds);
		for(int i=0; i<model.numFolds; ++i) {
    		CorrLDA corrlda = new CorrLDA(nitter, alpha, beta, gamma, model, i);
    		corrlda.estimate();
    		logLik.add(corrlda.calcLogLikelihood());
    	}
		String filename = "logLik_" + nitter + "_" + alpha + "_" + beta + "_" + gamma;
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
	                new FileOutputStream(dataDir + filename)));
			for(double ll: logLik)
				writer.write(ll + "\n");
			writer.flush(); 
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Finished thread " + threadID);
	}

}
