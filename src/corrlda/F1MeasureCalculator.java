package corrlda;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class F1MeasureCalculator {
	public static void main(String args[]) throws IOException {
		int nthreads = 10;
		if(args.length>0)
			nthreads = Integer.parseInt(args[0]);
		
    	ExecutorService executor = Executors.newFixedThreadPool(nthreads);
    	
		List<Future<Double>> list = new ArrayList<Future<Double>>();
		
//    	int[] iter = {10,20,50};
    	int[] iter = {20};
//		double[] alphas = {0.01, 0.05, 0.1, 0.5, 1, 1.5, 2};
    	double[] alphas = {0.01};
//    	double[] betas = {0.01, 0.05, 0.1, 0.5, 1, 1.5, 2};
    	double[] betas = {0.05};
//    	double[] gammas = {0.01, 0.05, 0.1, 0.5, 1, 1.5, 2};
    	double[] gammas = {1.5};
    	Model model = new Model();
    	model.initialize();
    	
    	for(int i: iter) {
	    	for(double a: alphas) {
	    		for(double b: betas) {
	    			for(double g: gammas) {
	    				for(int f=0; f<model.numFolds; ++f) {
		    				Callable<Double> worker = new FoldMultiThreadedRunner(model, a, b, g, i, f);
		    				Future<Double> submit = executor.submit(worker);
		    				list.add(submit); 
	    				}
	    			}
	    		}
	    	}
    	}
		// Now retrieve the result
		for (Future<Double> future : list) {
			try {
				System.out.println(future.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		// This will make the executor accept no new threads
		// and finish all existing threads in the queue
		executor.shutdown();
		
		System.out.println("Finished all threads");    	
    }
}
