package DPHC.DPHC2;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.util.Arrays;
import java.util.List;
import java.util.Random;


import DPHC.DPHC2.readmatfile.readmatfiles;
import DPHC.DPHC2.wavelet.Wavelet;

public class MainActivity extends Activity {

    private String message;
    private TextView mTextView;
    private Random random = new Random();

    private int pcaInputCalculator(int rawDownSample, int waveletDownSample, int waveletOmit) {

        int pcaInput = 0;

        if (rawDownSample == 1 && waveletDownSample == 1 && waveletOmit == 0) {
            pcaInput = 1026;
        } else if (rawDownSample == 1 && waveletDownSample == 1 && waveletOmit == 1) {
            pcaInput = 774;
        } else if (rawDownSample == 1 && waveletDownSample == 1 && waveletOmit == 2) {
            pcaInput = 646;
        } else if (rawDownSample == 1 && waveletDownSample == 2 && waveletOmit == 0) {
            pcaInput = 778;
        } else if (rawDownSample == 1 && waveletDownSample == 2 && waveletOmit == 1) {
            pcaInput = 650;
        } else if (rawDownSample == 1 && waveletDownSample == 2 && waveletOmit == 2) {
            pcaInput = 584;
        } else if (rawDownSample == 2 && waveletDownSample == 1 && waveletOmit == 0) {
            pcaInput = 776;
        } else if (rawDownSample == 2 && waveletDownSample == 1 && waveletOmit == 1) {
            pcaInput = 524;
        } else if (rawDownSample == 2 && waveletDownSample == 1 && waveletOmit == 2) {
            pcaInput = 396;
        } else if (rawDownSample == 2 && waveletDownSample == 2 && waveletOmit == 0) {
            pcaInput = 528;
        } else if (rawDownSample == 2 && waveletDownSample == 2 && waveletOmit == 1) {
            pcaInput = 400;
        } else if (rawDownSample == 2 && waveletDownSample == 2 && waveletOmit == 2) {
            pcaInput = 334;
        }

        return pcaInput;
    }

    private float[] dot(float[] in1, float[] in2) {

        if (in2.length == 1) {
            for (int i = 0; i < in1.length; i++) {
                in1[i] = in1[i] * in2[0];
            }
        } else {
            for (int i = 0; i < in1.length; i++) {
                in1[i] = in1[i] * in2[i];
            }
        }

        return in1;
    }

    private float[] sum3vector(float[] in1, float[] in2, float[] in3) {

        for (int i = 0; i < in1.length; i++) {
            in1[i] = in1[i] + in2[i] + in3[i];
        }

        return in1;
    }

    private float[] sum2Vector(float[] in1, float[] in2) {

        for (int i = 0; i < in1.length; i++) {
            in1[i] = in1[i] + in2[i];
        }

        return in1;
    }

    private float[] tanHEval(float[] in) {

        for (int i = 0; i < in.length; i++) {
            in[i] = (float) Math.tanh(in[i]);
        }

        return in;
    }

    private float[] sigmoid(float[] in) {
        for (int i = 0; i < in.length; i++) {
            in[i] = (float) (1 / (1 + Math.exp(-1 * in[i])));
        }
        return in;
    }

    private float[][] arrayCutter2D(float[][] input, int x, int y) {
        float[][] output = new float[x][y];

        for (int i = 0; i < x; i++) {
            System.arraycopy(input[i], 0, output[i], 0, y);
        }
        return output;
    }

    private float[] arrayCutter1D(float[] input, int x) {
        float[] output = new float[x];

        System.arraycopy(input, 0, output, 0, x);
        return output;
    }

    public float[][] transposeMatrix(float[][] m) {

        int rowNumber = m.length;
        int columnNumber = m[0].length;

        float[][] temp = new float[columnNumber][rowNumber];

        for (int i = 0; i < rowNumber; i++) {
            for (int j = 0; j < columnNumber; j++) {
                temp[j][i] = m[i][j];
            }
        }
        return temp;
    }

    private float[] newCrossInRange(float[] in1, float[][] in2, int j1, int j2) {

        int n = in2.length;
        float[] res = new float[n];

        for (int i = 0; i < n; i++) {
            for (int j = j1; j < j2; j++) {
                res[i] += in2[i][j] * in1[j];
            }
        }
        return res;
    }

    private float[] newCross(float[] in1, float[][] in2) {

        int n = in2.length;//600
        int m = in2[0].length;//1026

        float[] res = new float[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                res[i] += in2[i][j] * in1[j];
            }
        }
        return res;
    }

    private float[][] randomInit2D(float[][] in) {

        int dim1 = in.length;
        int dim2 = in[1].length;

        for (int i = 0; i < dim1; i++) {
            in[i] = new float[dim2];
        }

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                in[i][j] = random.nextFloat();
            }
        }
        return in;
    }

    private float[] appender(float[] x1, float[] x2, float[] x3, float[] x4, float[] x5,
                             float[] x6) {
        float[] output =
                new float[x1.length + x2.length + x3.length + x4.length + x5.length + x6.length];

        int index = 0;
        for (float temp : x1) {
            output[index] = temp;
            index++;
        }
        for (float temp : x2) {
            output[index] = temp;
            index++;
        }
        for (float temp : x3) {
            output[index] = temp;
            index++;
        }
        for (float temp : x4) {
            output[index] = temp;
            index++;
        }
        for (float temp : x5) {
            output[index] = temp;
            index++;
        }
        for (float temp : x6) {
            output[index] = temp;
            index++;
        }
        return output;
    }

    private float[] concatenate(float[] x1, float[] x2) {
        float[] output =
                new float[x1.length + x2.length];

        int index = 0;
        for (float temp : x1) {
            output[index] = temp;
            index++;
        }
        for (float temp : x2) {
            output[index] = temp;
            index++;
        }

        return output;
    }



    private float[][] randomArray(int n,int m) {
        float[][] randomArray = new float[n][m];
        Random randNumGenerator = new Random();

        for (int i = 0; i < n; i++) {
            for(int j=0; j < m ;j++) {

                randomArray[i][j] = random.nextFloat();

            }
        }
        return randomArray;
    }




    private float[] downSample(float[] x, int rate) {

        if (rate == 1)
            return x;
        else {
            float[] output = new float[x.length / 2];
            int index = 0;
            for (int i = 0; i < x.length; i += 2, index++) {
                output[index] = x[i];
            }
            return output;
        }
    }

    private void killBakGroundProcess() {

        Context context = getApplicationContext();
        List<ApplicationInfo> packages;
        PackageManager pm = getPackageManager();
        packages = pm.getInstalledApplications(0);

        ActivityManager mActivityManager = (ActivityManager) context.getSystemService(
                Context.ACTIVITY_SERVICE);
        String myPackage = getApplicationContext().getPackageName();

        if(mActivityManager != null){
            for (ApplicationInfo packageInfo : packages) {
                if ((packageInfo.flags & ApplicationInfo.FLAG_SYSTEM) == 1) continue;
                if (packageInfo.packageName.equals(myPackage)) continue;
                mActivityManager.killBackgroundProcesses(packageInfo.packageName);
            }
        }

    }

    private String mainFunction() throws Exception {

        killBakGroundProcess();

        double[] allPcaTime = new double[9];
        double[] allWaveletTime = new double[9];
        double[] allLstmTime = new double[9];
        double[] allLstm_A_Time = new double[9];
        double[] allblend_Time = new double[9];



        final int rawDownSample_B = 2;
        final int waveletDownSample_B = 2;
        final int waveletOmit_B = 0;
        final int pcaInputDim_B =  pcaInputCalculator(rawDownSample_B, waveletDownSample_B, waveletOmit_B);
        final int pcaOutputDim_B = 400;

        final int rawDownSample_A = 1;
        final int waveletDownSample_A = 2;
        final int waveletOmit_A = 0;
        final int pcaOutputDim_A = 0;




        final int lstmDepth = 5;
        final int lstmNh = 50;
        final int lstm_A_Nh=30;
        final int DepthA=10;

        final int lstmWidth = pcaOutputDim_B / lstmDepth;

/*          In order to get the app to work input matrices should be loaded here


            These inputs(w,u,b,c,h) are results of train data which should be generated in advance
*/
        //Weights
        readmatfiles matreader = new readmatfiles(getApplicationContext());
        float[][] w0 = transposeMatrix(matreader.get2DFloats("w0.txt",pcaOutputDim_B, lstmNh));


        float[][] w1 = transposeMatrix(matreader.get2DFloats("w1.txt",pcaOutputDim_B, lstmNh));
        float[][] w2 = transposeMatrix(matreader.get2DFloats("w2.txt",pcaOutputDim_B, lstmNh));
        float[][] w3 = transposeMatrix(matreader.get2DFloats("w3.txt",pcaOutputDim_B, lstmNh));

        float[][] u0 = transposeMatrix(matreader.get2DFloats("u0.txt", lstmNh, lstmNh));

        float[][] u1 = transposeMatrix(matreader.get2DFloats("u1.txt", lstmNh, lstmNh));

        float[][] u2 = transposeMatrix(matreader.get2DFloats("u2.txt", lstmNh, lstmNh));

        float[][] u3 = transposeMatrix(matreader.get2DFloats("u3.txt", lstmNh, lstmNh));

        //offsets
        float[] b0 = matreader.get1DFloats("b0.txt",lstmNh);
        float[] b1 = matreader.get1DFloats("b1.txt",lstmNh);
        float[] b2 = matreader.get1DFloats("b2.txt",lstmNh);
        float[] b3 = matreader.get1DFloats("b3.txt",lstmNh);



        float[] c = matreader.get1DFloats("c.txt",lstmNh);
        float[] h = matreader.get1DFloats("h.txt",lstmNh);
        float[] c_ecg = matreader.get1DFloats("c_ecg.txt",lstm_A_Nh);
        float[] h_ecg = matreader.get1DFloats("h_ecg.txt",lstm_A_Nh);
        float[] c_w = matreader.get1DFloats("c_w.txt",lstm_A_Nh);
        float[] h_w = matreader.get1DFloats("h_w.txt",lstm_A_Nh);

        final int dim_L=510;
        final int dim_R=280;

        final int lstmWidth_A1 =  dim_L / DepthA ;
        final int lstmWidth_A2 = dim_R / DepthA ;


        float[][] w0_A1 = transposeMatrix(matreader.get2DFloats("w0_A1.txt",dim_L, lstm_A_Nh));


        float[][] w1_A1 = transposeMatrix(matreader.get2DFloats("w1_A1.txt",dim_L, lstm_A_Nh));


        float[][] w2_A1 = transposeMatrix(matreader.get2DFloats("w2_A1.txt",dim_L, lstm_A_Nh));


        float[][] w3_A1 = transposeMatrix(matreader.get2DFloats("w3_A1.txt",dim_L, lstm_A_Nh));


        float[][] w0_A2 = transposeMatrix(matreader.get2DFloats("w0_A2.txt",dim_R, lstm_A_Nh));


        float[][] w1_A2 = transposeMatrix(matreader.get2DFloats("w1_A1.txt",dim_R, lstm_A_Nh));


        float[][] w2_A2 = transposeMatrix(matreader.get2DFloats("w2_A2.txt",dim_R, lstm_A_Nh));


        float[][] w3_A2 = transposeMatrix(matreader.get2DFloats("w3_A2.txt",dim_R, lstm_A_Nh));



        float[][] u0_A = transposeMatrix(matreader.get2DFloats("u0_A.txt",lstm_A_Nh, lstm_A_Nh));


        float[][] u1_A = transposeMatrix(matreader.get2DFloats("u1_A.txt",lstm_A_Nh, lstm_A_Nh));


        float[][] u2_A = transposeMatrix(matreader.get2DFloats("u2_A.txt",lstm_A_Nh, lstm_A_Nh));


        float[][] u3_A = transposeMatrix(matreader.get2DFloats("u3_A.txt",lstm_A_Nh, lstm_A_Nh));


        float[] b0_A = matreader.get1DFloats("b0_A.txt",lstm_A_Nh);
        float[] b1_A = matreader.get1DFloats("b1_A.txt",lstm_A_Nh);
        float[] b2_A = matreader.get1DFloats("b2_A.txt",lstm_A_Nh);
        float[] b3_A = matreader.get1DFloats("b3_A.txt",lstm_A_Nh);





        //fully connected
        float[][] fullyConnected = transposeMatrix(matreader.get2DFloats("fullyconnected.txt", lstmNh ,7));

        float[][] fullyConnected_A = transposeMatrix(matreader.get2DFloats("fullyconnected_A.txt", 2*lstm_A_Nh ,7));

        //first layer of blend
        float[][] fullyConnected_Blend_0 = transposeMatrix(matreader.get2DFloats("fullyconnected_Blend_0.txt",14,80));

        float[][] fullyConnected_Blend_1 = transposeMatrix(matreader.get2DFloats("fullyconnected_Blend_1.txt",80,10));

        //second layer of blend
        float[][] fullyConnected_Blend_2 = transposeMatrix(matreader.get2DFloats("fullyconnected_Blend_2.txt",10,7));


        float[] fullyConnectedB = matreader.get1DFloats("fullyconnectedB.txt",7);

        float[] fullyConnectedB_A = matreader.get1DFloats("fullyconnectedB_A.txt",7);


        float[] fullyConnectedB_Blend_0 = matreader.get1DFloats("fullyconnectedB_Blend_0.txt",80);

        float[] fullyConnectedB_Blend_1 = matreader.get1DFloats("fullyconnectedB_Blend_1.txt",10);



        float[] fullyConnectedB_Blend_2 = matreader.get1DFloats("fullyconnectedB_Blend_2.txt",7);




        //lstm input , these inputs should be interpreted from ecg input which is raw input of app

        float[] x = matreader.get1DFloats("x.txt",pcaOutputDim_B);
        float[] xr = matreader.get1DFloats("xr.txt",8);
        float[] x_ecg = matreader.get1DFloats("x_ecg.txt",500);
        float[] x_w = matreader.get1DFloats("x_w.txt",270);

        float[] firstRawInput = matreader.get1DFloats("firstRawInput.txt",260);
        float[] firstFeature = matreader.get1DFloats("firstFeature.txt",4);
        float[] secondRawInput = matreader.get1DFloats("secondRawInput.txt",260);
        float[] secondFeature = matreader.get1DFloats("secondFeature.txt",4);


        float[] highPassFilter = matreader.get1DFloats("high_pass_filter.txt",4);
        float[] lowPassFilter = matreader.get1DFloats("low_pass_filter.txt",4);

        float[][] tempPca = new float[pcaInputDim_B][pcaOutputDim_B];
        float[][] pca = transposeMatrix(randomInit2D(tempPca));

        float[] wavyInput1;
        float[] wavyInput2;
        float[] x1 = null;


        for (int index = 0; index < 9; index++) { //perform everything 10 timeds to get a reasonable time by taking median

            /*
             ********************************************************************
             *****************************   wavelet start  *********************
             ********************************************************************
             */
            double waveletStart = System.currentTimeMillis();

            Wavelet wavelet = new Wavelet();
            wavyInput1 = wavelet.wavelet(waveletOmit_B,
                    downSample(firstRawInput, waveletDownSample_B),
                    highPassFilter,
                    lowPassFilter);

            wavyInput2 = wavelet.wavelet(waveletOmit_B,
                    downSample(secondRawInput, waveletDownSample_B),
                    highPassFilter,
                    lowPassFilter);

            x1 = arrayCutter1D(appender(
                    downSample(firstRawInput, rawDownSample_B),
                    wavyInput1,
                    firstFeature,
                    downSample(secondRawInput, rawDownSample_B),
                    wavyInput2,
                    secondFeature), pcaInputDim_B);


            double waveletEnd = System.currentTimeMillis();
            double waveletTotalTime = waveletEnd - waveletStart;
            allWaveletTime[index] = waveletTotalTime;

            /*
             ********************************************************************
             *****************************   pca start  *************************
             ********************************************************************
             */
            long crossStartTime = System.currentTimeMillis();

            float[] betha_input=newCross(x1, pca);


            long crossEndTime = System.currentTimeMillis();
            allPcaTime[index] = crossEndTime - crossStartTime;

            /*
             ********************************************************************
             *****************************   Lstm start  ************************
             ********************************************************************
             */



            /*
                Model Betha
             */


            long lstmStart = System.currentTimeMillis();

            for (int l = 0; l < lstmDepth; l++) {

                c = sum2Vector(
                        dot(
                                sigmoid(sum3vector(//it
                                        newCrossInRange(betha_input, w1,
                                                l * lstmWidth, (l + 1) * lstmWidth),
                                        newCross(h, u1),
                                        b1)
                                ),
                                tanHEval(sum3vector(//mt
                                        newCrossInRange(betha_input, w0,
                                                l * lstmWidth, (l + 1) * lstmWidth),
                                        newCross(h, u0),
                                        b0)
                                )
                        ),
                        dot(
                                sigmoid(sum3vector(//ft
                                        newCrossInRange(betha_input, w2,
                                                l * lstmWidth, (l + 1) * lstmWidth),
                                        newCross(h, u2),
                                        b2)
                                ),
                                c
                        )
                );
                h = dot(
                        sigmoid(//ot
                                sum3vector(
                                        newCrossInRange(betha_input, w3, l * lstmWidth,
                                                (l + 1) * lstmWidth),
                                        newCross(h, u3),
                                        b3)
                        ),
                        tanHEval(c)
                );

            }

            //it's the fully connected layer

            float[] finalbetha = sum2Vector(newCross(h, fullyConnected), fullyConnectedB);

            long lstmEnd = System.currentTimeMillis();
            allLstmTime[index] = lstmEnd - lstmStart;
            //end of model betha




            /*
                Model alpha


             */

            float [] ttadd = {0,0};

            float[] xr_ecg=concatenate(xr,x_ecg);
            xr_ecg = concatenate(xr_ecg,ttadd);
            float[] xr_w=concatenate(xr,x_w);
            xr_w = concatenate(xr_w,ttadd);



            long lstm_A_Start = System.currentTimeMillis();

            for (int l = 0; l < DepthA; l++) {

                c_ecg = sum2Vector(
                        dot(
                                sigmoid(sum3vector(//it
                                        newCrossInRange(xr_ecg, w1_A1,
                                                l * lstmWidth_A1, (l + 1) * lstmWidth_A1),
                                        newCross(h_ecg, u1_A),
                                        b1_A)
                                ),
                                tanHEval(sum3vector(//mt
                                        newCrossInRange(xr_ecg, w0_A1,
                                                l * lstmWidth_A1, (l + 1) * lstmWidth_A1),
                                        newCross(h_ecg, u0_A),
                                        b0_A)
                                )
                        ),
                        dot(
                                sigmoid(sum3vector(//ft
                                        newCrossInRange(xr_ecg, w2_A1,
                                                l * lstmWidth_A1, (l + 1) * lstmWidth_A1),
                                        newCross(h_ecg, u2_A),
                                        b2_A)
                                ),
                                c_ecg
                        )
                );
                h_ecg = dot(
                        sigmoid(//ot
                                sum3vector(
                                        newCrossInRange(xr_ecg, w3_A1, l * lstmWidth_A1,
                                                (l + 1) * lstmWidth_A1),
                                        newCross(h_ecg, u3_A),
                                        b3_A)
                        ),
                        tanHEval(c_ecg)
                );

                c_w = sum2Vector(
                        dot(
                                sigmoid(sum3vector(//it
                                        newCrossInRange(xr_w, w1_A2,
                                                l * lstmWidth_A2, (l + 1) * lstmWidth_A2),
                                        newCross(h_w, u1_A),
                                        b1_A)
                                ),
                                tanHEval(sum3vector(//mt
                                        newCrossInRange(xr_w, w0_A2,
                                                l * lstmWidth_A2, (l + 1) * lstmWidth_A2),
                                        newCross(h_w, u0_A),
                                        b0_A)
                                )
                        ),
                        dot(
                                sigmoid(sum3vector(//ft
                                        newCrossInRange(xr_w, w2_A2,
                                                l * lstmWidth_A2, (l + 1) * lstmWidth_A2),
                                        newCross(h_w, u2_A),
                                        b2_A)
                                ),
                                c_w
                        )
                );
                h_w = dot(
                        sigmoid(//ot
                                sum3vector(
                                        newCrossInRange(xr_w, w3_A2, l * lstmWidth_A2,
                                                (l + 1) * lstmWidth_A2),
                                        newCross(h_w, u3_A),
                                        b3_A)
                        ),
                        tanHEval(c_w)
                );

            }

            //it's the fully connected layer

            float[] h_final=concatenate(h_ecg,h_w);

            float [] finalalpha=sum2Vector(newCross(h_final, fullyConnected_A), fullyConnectedB_A);

            long lstm_A_End = System.currentTimeMillis();
            allLstm_A_Time[index] = lstm_A_End - lstm_A_Start;

            //end of model alpha





            /*
                Blender

             */

            long blendStart = System.currentTimeMillis();
            float [] finalVector= concatenate(finalalpha,finalbetha);

            float [] blend0=sum2Vector(newCross(finalVector,fullyConnected_Blend_0),fullyConnectedB_Blend_0); // to 80 output layer

            float [] blend=sum2Vector(newCross(blend0,fullyConnected_Blend_1),fullyConnectedB_Blend_1);  // to 10 output layer

            float [] result=sum2Vector(newCross(blend,fullyConnected_Blend_2),fullyConnectedB_Blend_2);

            long blend_End = System.currentTimeMillis();
            allblend_Time[index] = blend_End - blendStart;



            try {
                Thread.sleep(6000); //sleep time to let the device cool down,configure depending on device
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }

        Arrays.sort(allWaveletTime);
        Arrays.sort(allPcaTime);
        Arrays.sort(allLstmTime);
        Arrays.sort(allLstm_A_Time);
        Arrays.sort(allblend_Time);

        double totalTime = allWaveletTime[4] + allPcaTime[4] + allLstmTime[4] + allLstm_A_Time[4] + allblend_Time[4];
        return "execution time is: " + totalTime + "per task is: " + allWaveletTime[4] + "," +
                allPcaTime[4] + "," + allLstmTime[4] + "," + allLstm_A_Time[4] + "," + allblend_Time[4] ;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.round_activity_main);
        mTextView = findViewById(R.id.text);
        String welcomeString = "welcome to Deep Personal Heart Care(DPHC)";
        mTextView.setText(welcomeString);

//        Thread backgroundThread = new Thread(new Runnable() {
//            @Override
//            public void run() {
//                message = mainFunction();
//            }
//        });
//
//        try {
//
//            backgroundThread.start();
//            backgroundThread.join();
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }


        MainFunctionTask mainFunctionTask = new MainFunctionTask();
        mainFunctionTask.execute();


//        String wholeMessage = mTextView.getText() + "\n" +  message;
//        mTextView.setText(wholeMessage);
//        Log.d("TIME", "Total execution time with Thread is: " + message);
    }

    private class MainFunctionTask extends AsyncTask<Void, Void, String> {

        @Override
        protected String doInBackground(Void... params) {
                try {
                    return mainFunction();
                }catch(Exception e){
                    e.printStackTrace();
                    return "file not found";

                }
        }

        @Override
        protected void onPostExecute(String message) {
            Log.d("Time", "The total time with AsyncTask is:" + message);
            mTextView.setText(mTextView.getText() + "\n" + message);
        }
    }


}

