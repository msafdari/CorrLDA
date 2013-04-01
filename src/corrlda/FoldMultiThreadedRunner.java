package corrlda;

import java.util.concurrent.Callable;

public class FoldMultiThreadedRunner implements Callable<Double>{
		
	private static int threadCounter = 0;
	
	private double alpha;
	private double beta;
	private double gamma;
	private int nitter;
	private int threadID;
	private Model model;
	private int foldIdx;
	
	public FoldMultiThreadedRunner(Model m) {
		model = m;
		alpha = 1.0;
		beta = 2.0;
		gamma = 1.0;
		nitter = 10;
		threadID = threadCounter++;
	}
	
	public FoldMultiThreadedRunner(Model m, double a, double b, double g, int iter, int f) {
		this(m);
		alpha = a;
		beta = b;
		gamma = g;
		nitter = iter;
		foldIdx = f;
	}


	@Override
	public Double call() throws Exception {
		// TODO Auto-generated method stub
		
//		System.out.println("Starting thread " + threadID);
		CorrLDA corrlda = new CorrLDA(nitter, alpha, beta, gamma, model, foldIdx);
		corrlda.estimate();
		double f1 = corrlda.getF1Measure();
		
//		System.out.println("Ending thread " + threadID);
		return f1;
	}

}
