/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package corrlda;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;

/**
 *
 * @author kaldr
 */
public class Estimate {

    public void estimate(CorrLDA corrlda) {
//        AUC auc=new AUC();
        int topic = 0;
        int ec = 0;
        System.out.println("***************************\nSampling " + corrlda.nitter + " iteration!\n***************************");
        for (int iter = 1; iter < corrlda.nitter; ++iter) {
            long startTime = System.currentTimeMillis();
            System.out.println("Iteration " + (iter+1) + " ...");
            for (int imageID = 0; imageID < corrlda.numTrainImages; ++imageID) {
            	int numVisualTokens = corrlda.model.visualWordsInImages.get(imageID).size();
                for (int visualToken = 0; visualToken < numVisualTokens; ++visualToken) {
                    topic = samplingMovieTopic(imageID, visualToken, corrlda);
                    corrlda.z[imageID].set(visualToken, topic);
                }
                for (int tagToken = 0; tagToken < corrlda.model.annotWordsInImages.get(imageID).size(); ++tagToken) {
                    topic = samplingTagTopic(imageID, tagToken, numVisualTokens, corrlda);
                    corrlda.ztag[imageID].set(tagToken, topic);
                }
            }
            ec = ec + 1;
            long endTime = System.currentTimeMillis();
            double time = 0;
            time = time + (endTime - startTime) / 1000;
            double timeover = time * (corrlda.nitter - ec) / 60;
            System.out.println("Ecalepsed time: " + (time) + "s, Completed in " + timeover + " m");
        }
        System.out.println("Gibbs sampling completed!\n");
        System.out.println("Saving the final model!\n");
        computeTheta(corrlda, corrlda.nitter - 1);
        computePhi(corrlda, corrlda.nitter - 1);
        computeDigamma(corrlda, corrlda.nitter - 1);        
        //saveModel(corrlda, iter - 1 + no);
//        System.out.println(auc.computeAUC(corrlda));
    }

    public double calcLogLikelihood(CorrLDA corrlda, int[] testSetIdx) {
        int nitter = corrlda.nitter;
        double likelihood = 0.0;
        for (int imageID : testSetIdx) {
        	ArrayList<Integer> featureVector = corrlda.model.visualWordsInImages.get(imageID);
        	ArrayList<Integer> annotVector = corrlda.model.annotWordsInImages.get(imageID);
            for(int visualToken = 0; visualToken < featureVector.size(); ++visualToken) {
            	int topic = corrlda.z[imageID].get(visualToken);
            	likelihood += Math.log(corrlda.theta[imageID][topic] * corrlda.phi[featureVector.get(visualToken)][topic]);
            }
            for(int tagToken = 0; tagToken < annotVector.size(); ++tagToken) {
            	int topic = corrlda.ztag[imageID].get(tagToken);
            	likelihood += Math.log(corrlda.digamma[annotVector.get(tagToken)][topic] / featureVector.size());
            }
        }
        System.out.println(likelihood);

        return likelihood;
    }

    public void saveTheta(CorrLDA corrlda, int iter) {
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
            writer.write(corrlda.numTrainImages + " " + corrlda.K + "\r\n");
            writer.flush();
            for (int u = 0; u < corrlda.numTrainImages; u++) {
                for (int k = 0; k < corrlda.K; k++) {
                    String t = Double.toString(corrlda.theta[u][k]);
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

    public void savePhi(CorrLDA corrlda, int iter) {
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
            writer.write(corrlda.visualVocabLen + " " + corrlda.K + "\r\n");
            writer.flush();
            for (int u = 0; u < corrlda.visualVocabLen; u++) {
                for (int k = 0; k < corrlda.K; k++) {
                    String t = Double.toString(corrlda.phi[u][k]);
                    writer.write(t + " ");
                    writer.flush();

                }
                writer.write("\r\n");
                writer.flush();
            }
        } catch (Exception e) {
        }
    }

    public void saveDigamma(CorrLDA corrlda, int iter) {
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
            writer.write(corrlda.annotVocabLen + " " + corrlda.K + "\r\n");
            writer.flush();
            for (int u = 0; u < corrlda.annotVocabLen; u++) {
                for (int k = 0; k < corrlda.K; k++) {
                    String t = Double.toString(corrlda.digamma[u][k]);
                    writer.write(t + " ");
                    writer.flush();
                }
                writer.write("\r\n");
                writer.flush();
            }
        } catch (Exception e) {
        }
    }

    public int samplingMovieTopic(int imageID, int visualToken, CorrLDA corrlda) {
        int topic = corrlda.z[imageID].get(visualToken);
        int visualWordID = corrlda.model.visualWordsInImages.get(imageID).get(visualToken);
        corrlda.nv_z[visualWordID][topic] -= 1;
        corrlda.nsumv_z[topic] -= 1;
        corrlda.nz_d[imageID][topic] -= 1;
        corrlda.nsumz_d[imageID] -= 1;

        double Vbeta = corrlda.visualVocabLen * corrlda.beta;
        double Kalpha = corrlda.K * corrlda.alpha;
        double[] p = new double[corrlda.K];
        for (int k = 0; k < corrlda.K; k++) {
            p[k] = (corrlda.nv_z[visualWordID][k] + corrlda.beta) / (corrlda.nsumv_z[k] + Vbeta)
                    * (corrlda.nz_d[imageID][k] + corrlda.alpha) / (corrlda.nsumz_d[imageID] + Kalpha);
        }
        for (int k = 1; k < corrlda.K; k++) {
            p[k] += p[k - 1];
        }
        double sample = Math.random() * p[corrlda.K - 1];
        for (topic = 0; topic < corrlda.K; topic++) {
            if (p[topic] > sample) {
                break;
            }
        }
        if (topic == corrlda.K) {
            topic -= 1;
        }
        corrlda.nv_z[visualWordID][topic] += 1;
        corrlda.nsumv_z[topic] += 1;
        corrlda.nz_d[imageID][topic] += 1;
        corrlda.nsumz_d[imageID] += 1;
        return topic;
    }

    public int samplingTagTopic(int imageID, int tagToken, int numVisualTokens, CorrLDA corrlda) {
        int topic = corrlda.ztag[imageID].get(tagToken);//*********************************************
        int tagID = corrlda.model.annotWordsInImages.get(imageID).get(tagToken);
        corrlda.nt_z[tagID][topic] -= 1;
        corrlda.nsumt_z[topic] -= 1;
        double Tgamma = corrlda.annotVocabLen * corrlda.gamma;
        double[] p = new double[corrlda.K];
        for (int k = 0; k < corrlda.K; k++) {
            p[k] = corrlda.nz_d[imageID][k] / ((double) numVisualTokens) * (corrlda.nt_z[tagID][k] + corrlda.gamma) / (corrlda.nsumt_z[k] + Tgamma);
        }

        for (int k = 1; k < corrlda.K; k++) {
            p[k] += p[k - 1];
        }
        double sample = Math.random() * p[corrlda.K - 1];

        for (topic = 0; topic < corrlda.K; topic++) {
            if (p[topic] > sample) {
                break;
            }
        }
        if (topic == corrlda.K) {
            topic -= 1;
        }
        corrlda.nt_z[tagID][topic] += 1;
        corrlda.nsumt_z[topic] += 1;
        return topic;
    }

    public void computePhi(CorrLDA corrlda, int iter) {
        for (int m = 0; m < corrlda.visualVocabLen; m++) {
            for (int k = 0; k < corrlda.K; k++) {
                corrlda.phi[m][k] = (corrlda.nv_z[m][k] + corrlda.beta) / (corrlda.nsumv_z[k] + corrlda.visualVocabLen * corrlda.beta);
            }

        }
        for (int k = 0; k < corrlda.K; k++) {
            double sum=0;
            for (int t = 0; t < corrlda.visualVocabLen; t++) {
                sum+=corrlda.phi[t][k];
            }
            for (int t = 0; t < corrlda.visualVocabLen; t++) {
                corrlda.phi[t][k]/=sum;
            }
        }
        savePhi(corrlda, iter);
    }

    public void computeTheta(CorrLDA corrlda, int iter) {
        for (int u = 0; u < corrlda.numTrainImages; ++u) {
            double sum = 0;
            for (int k = 0; k < corrlda.K; ++k) {
                corrlda.theta[u][k] = (corrlda.nz_d[u][k] + corrlda.alpha) / (corrlda.nsumz_d[u] + corrlda.K * corrlda.alpha);
                sum += corrlda.theta[u][k];
            }
            
            for (int k = 0; k < corrlda.K; k++) {
                corrlda.theta[u][k] /= sum;
            }
             

        }
        saveTheta(corrlda, iter);
    }

    public void computeDigamma(CorrLDA corrlda, int iter) {

        for (int t = 0; t < corrlda.annotVocabLen; t++) {

            for (int k = 0; k < corrlda.K; k++) {
                corrlda.digamma[t][k] = (corrlda.nt_z[t][k] + corrlda.gamma) / (corrlda.nsumt_z[k] + corrlda.annotVocabLen * corrlda.gamma);
            }
        }
        
        for (int k = 0; k < corrlda.K; k++) {
            double sum=0;
            for (int t = 0; t < corrlda.annotVocabLen; t++) {
                sum+=corrlda.digamma[t][k];
            }
            for (int t = 0; t < corrlda.annotVocabLen; t++) {
                corrlda.digamma[t][k]/=sum;
            }
        }
        saveDigamma(corrlda, iter);
    }
}
