/*
 * This programme is Correlative LDA 1
 * The original model is in the paper: Statistical entity-topic models
 */
package corrlda;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.TreeSet;
import java.util.Vector;

/**
 *
 * @author kaldr
 */
public class CorrLDA{
    /*
     * Model requirement
     */
    //Parameters

    public Integer K = 3;//k is the topic number;
    public double alpha = 1;
    public double beta = 2;
    public double gamma = 1;
    public double[][] theta;//theta is the topic distribution for each image; theta=p(z|image); theta~Dir(alpha)
    public Vector<Integer>[] z;//z is the topic chosen for a movie; z~mul(theta)
    public Vector<Integer>[] ztag;//ztag is the topic chosen for a code value; ztag~uni(z). tag = annotation
    public double[][] phi;//fine is p(visual word|z); phi~Dir(beta)
    public double[][] digamma;//diggamma is p(tag|ztag); diggama~Dir(gamma)
    public double Vbeta, Kalpha, Tgamma;
    //Counts
    public int visualVocabLen;
    public int numTrainImages;
    public int numTestImages;
    public int annotVocabLen;
    public int[][] nz_d;//number of times that topic z has occured in image d;
    public int[][] nv_z;//number of times the visual word is assigned to topic z;
    public int[][] nt_z;//number of times tag t is generated from topic z;
    public int[] nsumz_d;//sum of all the topics occured in image d;
    public int[] nsumv_z;//sum of all visual words assigned to topic k;
    public int[] nsumt_z;//sum of all the tags assigned to topic k;
    //configuration
    public int nitter = 3;
    public Model model;
    
    public ArrayList<Integer> trainSetIdx;
    public ArrayList<Integer> testSetIdx;
    
    private static class TagWordF1 {
		int tagID;
		double f1Measure;
		
		public TagWordF1(int t, double f) {
			tagID = t;
			f1Measure = f;
		}
	}
	
	public class TagWordF1Comparator implements Comparator<TagWordF1> {
	    @Override
	    public int compare(TagWordF1 x, TagWordF1 y) {
	        if (x.f1Measure > y.f1Measure)
	            return -1;
	        if (x.f1Measure < y.f1Measure)
	            return 1;
	        return 0;
	    }
	}
    
    public CorrLDA() {}

    public CorrLDA(Model m, int foldIdx){
        model = m;
        trainSetIdx = model.trainingFolds.get(foldIdx);
        testSetIdx = model.testingFolds.get(foldIdx);
        numTrainImages = trainSetIdx.size();
        numTestImages = testSetIdx.size();
        visualVocabLen = model.visualVocabSize;
        annotVocabLen = model.annotVocabSize;
        theta = new double[numTrainImages][K];
        phi = new double[visualVocabLen][K];
        digamma = new double[annotVocabLen][K];
        Vbeta = visualVocabLen * beta;
        Kalpha = K * alpha;
        Tgamma = annotVocabLen * gamma;
    }
    
    public CorrLDA(int iter, double a, double b, double g, Model m, int foldIdx) {
    	this(m, foldIdx);
    	alpha = a;
    	beta = b;
    	gamma = g;
    	Vbeta = visualVocabLen * beta;
        Kalpha = K * alpha;
        Tgamma = annotVocabLen * gamma;
        nitter = iter;
    }

    public void initialize(boolean testMode){
    	int numImages = testMode? numTestImages: numTrainImages;
        nz_d = new int[numImages][K];
        nsumz_d = new int[numImages];
        if(!testMode) {
        	nv_z = new int[visualVocabLen][K];
        	nt_z = new int[annotVocabLen][K];
        	nsumv_z = new int[K];
            nsumt_z = new int[K];
        }        

        //initial topic/z for movies and tags;
        z = new Vector[numImages];
        ztag = new Vector[numImages];
        for (int d = 0; d < numImages; ++d) {
        	int imageID = testMode? testSetIdx.get(d): trainSetIdx.get(d);
            ArrayList<Integer> featureVector = model.visualWordsInImages.get(imageID);
            int numVisualTokens = featureVector.size();
            z[d] = new Vector();
            ztag[d] = new Vector();

            for (int visualWordID: featureVector) {
                int topic = (int) (Math.random() * K);                    
                z[d].add(topic);
                //number of topic occured in this image
                nz_d[d][topic] += 1;
                nsumz_d[d] += 1;
                if(!testMode) {
	                //number of words assigned to topic
	                nv_z[visualWordID][topic] += 1;
	                nsumv_z[topic] += 1;
                }
            }
            
            ArrayList<Integer> annotVector = model.annotWordsInImages.get(imageID);
            for (int tagWordID: annotVector) {
                int zID = (int) Math.floor(Math.random() * numVisualTokens);
                int topic = z[d].get(zID);
                ztag[d].add(topic);
                if(!testMode) {
	                //number of tag assigend to topic
	                nt_z[tagWordID][topic] += 1;
	                nsumt_z[topic] += 1;
                }
            }
        }
    }
    
    public void estimate() {
    	estimate (false); // training set
    	estimate (true); // test set
    }
    
    public double getF1Measure() {
    	double sum = 0.0;
    	for(int i=0; i<testSetIdx.size(); ++i) {
    		sum += getF1Measure(i, testSetIdx.get(i));
    	}
    	return sum / testSetIdx.size();
    }
    
    public double getF1Measure(int testIdx, int testImageID) {
    	Comparator<TagWordF1> comp = new TagWordF1Comparator();
		PriorityQueue<TagWordF1> pqueue = new PriorityQueue<TagWordF1>(annotVocabLen, comp);
		double[][] pz_w = new double[K][visualVocabLen]; // p(z|w) = p(w|z)*p(z|d) / sum_over_all_topics(p(w|z)*p(z|d))
		for(int visualWordID: model.visualWordsInImages.get(testImageID)) {
			double sum = 0.0;
			for(int k=0; k<K; ++k) {
				pz_w[k][visualWordID] = phi[visualWordID][k] * theta[testIdx][k];
				sum += pz_w[k][visualWordID];
			}
			for(int k=0; k<K; ++k)
				pz_w[k][visualWordID] /= sum;
		}
		
		ArrayList<Integer> annotVector = model.annotWordsInImages.get(testImageID);
		TreeSet<Integer> testImageTags = new TreeSet<Integer>(annotVector);
		
//		for(int tagID: annotVector) {
		for(int tagID=0; tagID<model.annotVocabSize; ++tagID) {
			double prob = 0.0;
			for(int visualWordID: model.visualWordsInImages.get(testImageID)) {
				for(int k=0; k<K; ++k) {
					prob += pz_w[k][visualWordID] * digamma[tagID][k];
				}
			}
			TagWordF1 tagF1 = new TagWordF1(tagID, prob);
			pqueue.add(tagF1);			
		}
		
		double tp=0.0, fp=0.0, fn=0.0;
		for(int i=0; i<model.numTagsToRetrieve; ++i) {
			int guessedTag = pqueue.remove().tagID;
			if(testImageTags.contains(guessedTag)) {
				++tp;
				testImageTags.remove(guessedTag);
			}
			else {
				++fp;
			}
		}
		
		fn = testImageTags.size();
		
		double precision = tp / (tp+fp);
//		double recall = tp / (tp+fn);
//		return 2*precision*recall/(precision+recall);
		return precision;
    }
    
    private void estimate(boolean testMode) {
//      AUC auc=new AUC();
      initialize(testMode);
      int numImages = testMode? numTestImages: numTrainImages;
      for (int iter = 0; iter < nitter; ++iter) {
          for (int d = 0; d < numImages; ++d) {
        	int imageID = testMode? testSetIdx.get(d): trainSetIdx.get(d);  
          	int numVisualTokens = model.visualWordsInImages.get(imageID).size();
              for (int visualToken = 0; visualToken < numVisualTokens; ++visualToken) {
                  z[d].set(visualToken, samplingVisualWordTopic(d, imageID, visualToken, testMode));
              }
              for (int tagToken = 0; tagToken < model.annotWordsInImages.get(imageID).size(); ++tagToken) {
                  ztag[d].set(tagToken, samplingTagTopic(d, imageID, tagToken, numVisualTokens, testMode));
              }
          }
      }
      if(testMode) {
    	  computeTheta(nitter - 1, testMode);	      
      }
      else {
    	  computePhi(nitter - 1);
	      computeDigamma(nitter - 1);
      }
      //saveModel(corrlda, iter - 1 + no);
//      System.out.println(auc.computeAUC(corrlda));
  }

  public double calcLogLikelihood() {
      double likelihood = 0.0;
      for (int i=0; i<testSetIdx.size(); ++i) {
    	int imageID = testSetIdx.get(i);  
      	ArrayList<Integer> featureVector = model.visualWordsInImages.get(imageID);
      	ArrayList<Integer> annotVector = model.annotWordsInImages.get(imageID);
          for(int visualToken = 0; visualToken < featureVector.size(); ++visualToken) {
          	int topic = z[i].get(visualToken);
          	likelihood += Math.log(theta[i][topic] * phi[featureVector.get(visualToken)][topic]);
          }
          for(int tagToken = 0; tagToken < annotVector.size(); ++tagToken) {
          	int topic = ztag[i].get(tagToken);
          	likelihood += Math.log(digamma[annotVector.get(tagToken)][topic] / featureVector.size());
          }
      }
      return likelihood;
  }

  public void saveTheta(int iter) {
      String file = "Theta.dat";
      String pendix = Integer.toString(iter);
      if (iter < 10) {
          pendix = "000" + Integer.toString(iter);
      } else if (iter < 100) {
          pendix = "00" + Integer.toString(iter);
      } else if (iter < 1000) {
          pendix = "0" + Integer.toString(iter);
      }
      file = file + "_" + pendix;
      System.out.println("Saving theta...");
      try {
          BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                  new FileOutputStream(file), "UTF-8"));
          writer.write(numTrainImages + " " + K + "\r\n");
          writer.flush();
          for (int u = 0; u < numTrainImages; u++) {
              for (int k = 0; k < K; k++) {
                  String t = Double.toString(theta[u][k]);
                  writer.write(t + " ");
                  writer.flush();
              }
              writer.write("\r\n");
              writer.flush();
          }
      } catch (Exception e) {
          e.printStackTrace();
      }
  }

  public void savePhi(int iter) {
      System.out.println("Saving phi...");
      try {
          String file = "Phi.dat";
          String pendix = Integer.toString(iter);
          if (iter < 10) {
              pendix = "000" + Integer.toString(iter);
          } else if (iter < 100) {
              pendix = "00" + Integer.toString(iter);
          } else if (iter < 1000) {
              pendix = "0" + Integer.toString(iter);
          }
          file = file + "_" + pendix;
          BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                  new FileOutputStream(file), "UTF-8"));
          writer.write(visualVocabLen + " " + K + "\r\n");
          writer.flush();
          for (int u = 0; u < visualVocabLen; u++) {
              for (int k = 0; k < K; k++) {
                  String t = Double.toString(phi[u][k]);
                  writer.write(t + " ");
                  writer.flush();

              }
              writer.write("\r\n");
              writer.flush();
          }
      } catch (Exception e) {
      }
  }

  public void saveDigamma(int iter) {
      try {
          String file = "Digamma.dat";
          String pendix = Integer.toString(iter);
          if (iter < 10) {
              pendix = "000" + Integer.toString(iter);
          } else if (iter < 100) {
              pendix = "00" + Integer.toString(iter);
          } else if (iter < 1000) {
              pendix = "0" + Integer.toString(iter);
          }
          file = file + "_" + pendix;
          System.out.println("Saving digamma...");
          BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                  new FileOutputStream(file), "UTF-8"));
          writer.write(annotVocabLen + " " + K + "\r\n");
          writer.flush();
          for (int u = 0; u < annotVocabLen; u++) {
              for (int k = 0; k < K; k++) {
                  String t = Double.toString(digamma[u][k]);
                  writer.write(t + " ");
                  writer.flush();
              }
              writer.write("\r\n");
              writer.flush();
          }
      } catch (Exception e) {
      }
  }

  public int samplingVisualWordTopic(int docIdx, int imageID, int visualToken, boolean testMode) {
      int topic = z[docIdx].get(visualToken);
      int visualWordID = model.visualWordsInImages.get(imageID).get(visualToken);
      nz_d[docIdx][topic] -= 1;
      nsumz_d[docIdx] -= 1;
      if(!testMode) {
    	  nv_z[visualWordID][topic] -= 1;
          nsumv_z[topic] -= 1;
      }

      double[] p = new double[K];
      for (int k = 0; k < K; k++) {
          p[k] = (nv_z[visualWordID][k] + beta) / (nsumv_z[k] + Vbeta)
                  * (nz_d[docIdx][k] + alpha) / (nsumz_d[docIdx] + Kalpha);
      }
      for (int k = 1; k < K; k++) {
          p[k] += p[k - 1];
      }
      double sample = Math.random() * p[K - 1];
      for (topic = 0; topic < K; topic++) {
          if (p[topic] > sample) {
              break;
          }
      }
      if (topic == K)
          topic -= 1;
      if(!testMode) {
	      nv_z[visualWordID][topic] += 1;
	      nsumv_z[topic] += 1;
      }
      nz_d[docIdx][topic] += 1;
      nsumz_d[docIdx] += 1;
      return topic;
  }

  public int samplingTagTopic(int docIdx, int imageID, int tagToken, int numVisualTokens, boolean testMode) {
      int topic = ztag[docIdx].get(tagToken);//*********************************************
      int tagID = model.annotWordsInImages.get(imageID).get(tagToken);
      if(!testMode) {
	      nt_z[tagID][topic] -= 1;
	      nsumt_z[topic] -= 1;
      }
      double[] p = new double[K];
      for (int k = 0; k < K; k++) {
          p[k] = nz_d[docIdx][k] / ((double) numVisualTokens) * (nt_z[tagID][k] + gamma) / (nsumt_z[k] + Tgamma);
      }

      for (int k = 1; k < K; k++) {
          p[k] += p[k - 1];
      }
      double sample = Math.random() * p[K - 1];

      for (topic = 0; topic < K; topic++) {
          if (p[topic] > sample) {
              break;
          }
      }
      if (topic == K)
          topic -= 1;
      if(!testMode) {
	      nt_z[tagID][topic] += 1;
	      nsumt_z[topic] += 1;
      }
      return topic;
  }

  public void computePhi(int iter) {
      for (int m = 0; m < visualVocabLen; m++) {
          for (int k = 0; k < K; k++) {
              phi[m][k] = (nv_z[m][k] + beta) / (nsumv_z[k] + visualVocabLen * beta);
          }

      }
      for (int k = 0; k < K; k++) {
          double sum=0;
          for (int t = 0; t < visualVocabLen; t++) {
              sum+=phi[t][k];
          }
          for (int t = 0; t < visualVocabLen; t++) {
              phi[t][k]/=sum;
          }
      }
//      savePhi(iter);
  }

  public void computeTheta(int iter, boolean testMode) {
	  int numImages = testMode? numTestImages: numTrainImages;
      for (int u = 0; u < numImages; ++u) {
          double sum = 0;
          for (int k = 0; k < K; ++k) {
              theta[u][k] = (nz_d[u][k] + alpha) / (nsumz_d[u] + K * alpha);
              sum += theta[u][k];
          }
          
          for (int k = 0; k < K; k++) {
              theta[u][k] /= sum;
          }
           

      }
//      saveTheta(iter);
  }

  public void computeDigamma(int iter) {

      for (int t = 0; t < annotVocabLen; t++) {

          for (int k = 0; k < K; k++) {
              digamma[t][k] = (nt_z[t][k] + gamma) / (nsumt_z[k] + annotVocabLen * gamma);
          }
      }
      
      for (int k = 0; k < K; k++) {
          double sum=0;
          for (int t = 0; t < annotVocabLen; t++) {
              sum+=digamma[t][k];
          }
          for (int t = 0; t < annotVocabLen; t++) {
              digamma[t][k]/=sum;
          }
      }
//      saveDigamma(iter);
  }
}
