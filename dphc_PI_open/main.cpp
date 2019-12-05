#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <ctime>
#include <algorithm>

using namespace std;

vector<float> convolution1D(vector<float> inputSignal, vector<float> kernel);
vector<float> makeMirrorPadding(vector<float> &inputSignal, int mirrorPaddingSizeForBeginningOfArray, int mirrorPaddingSizeForEndOfArray);
vector<float> reverser(vector<float> &input);
float multiplier(vector<float> &inputSignal, vector<float> &kernel, int startIndexOfInput);
vector<float> wavelet(int omit, vector<float> &inputSignal, vector<float> &highPassFilter, vector<float> &lowPassFilter);
int pcaInputCalculator(int rawDownSample, int waveletDownSample, int waveletOmit);
vector<float> dot(vector<float> in1,const vector<float> &in2);
vector<float> sum3vector(vector<float> in1,const vector<float> &in2,const vector<float> &in3);
vector<float> sum2Vector(vector<float> in1,const vector<float> &in2);
vector<float> tanHEval(vector<float> in);
vector<float> sigmoid(vector<float> in);
void transposeMatrix(vector<vector<float> > &b);
vector<float> newCrossInRange(const vector<float> &in1,const vector<vector<float> > &in2, int j1, int j2);
vector<float> newCross(const vector<float> &in1,const vector<vector<float> > &in2);
vector<float> appender(vector<float> &x1, vector<float> &x2,vector<float> &x3, vector<float> &x4,vector<float> &x5, vector<float> &x6);
vector<float> concatenate(vector<float> &x1, vector<float> &x2);
vector<float> downSample(vector<float> &x,int rate);
vector<float> vectorcopy(vector<float> &a, int b0 , int b1);
vector<vector<float> > parse2dCSV(string filename,int m,int n);
vector<float> parse1dCSV(string filename,int n);




int main()
{

    float allPcaTime[9] = {};
    float allWaveletTime[9] = {};
    float allLstmTime[9] = {};
    float allLstm_A_Time[9] = {};
    float allblend_Time[9] = {};
    float lstmtime;

    int rawDownSample_B = 2;
    int waveletDownSample_B = 2;
    int waveletOmit_B = 0;
    int pcaInputDim_B =  pcaInputCalculator(rawDownSample_B, waveletDownSample_B, waveletOmit_B);
    int pcaOutputDim_B = 400;

    int lstmDepth = 5;
    int lstmNh = 50;
    int lstm_A_Nh=30;
    int DepthA=10;

    int lstmWidth = pcaOutputDim_B / lstmDepth;

/*          In order to get the code to work input matrices should be loaded here


            These inputs(w,u,b,c,h) are results of train data which should be generated in advance
*/
        //Weights for betha model

        vector<vector<float> > w0 =  (parse2dCSV("assets/w0.txt",pcaOutputDim_B, lstmNh));
        transposeMatrix(w0);

        vector<vector<float> > w1 = parse2dCSV("assets/w1.txt",pcaOutputDim_B, lstmNh);
        transposeMatrix(w1);
        vector<vector<float> > w2 =  (parse2dCSV("assets/w2.txt",pcaOutputDim_B, lstmNh));
        transposeMatrix(w2);

        vector<vector<float> > w3 =  (parse2dCSV("assets/w3.txt",pcaOutputDim_B, lstmNh));
        transposeMatrix(w3);

        vector<vector<float> > u0 =  (parse2dCSV("assets/u0.txt", lstmNh, lstmNh));
        transposeMatrix(u0);

        vector<vector<float> > u1 =  (parse2dCSV("assets/u1.txt", lstmNh, lstmNh));
        transposeMatrix(u1);

        vector<vector<float> > u2 =  (parse2dCSV("assets/u2.txt", lstmNh, lstmNh));
        transposeMatrix(u2);

        vector<vector<float> > u3 =  (parse2dCSV("assets/u3.txt", lstmNh, lstmNh));
        transposeMatrix(u3);

        //offsets
        vector<float> b0 = parse1dCSV("assets/b0.txt",lstmNh);

        vector<float> b1 =  parse1dCSV("assets/b1.txt",lstmNh);

        vector<float> b2 =  parse1dCSV("assets/b2.txt",lstmNh);

        vector<float> b3 =  parse1dCSV("assets/b3.txt",lstmNh);

        vector<float> c =  parse1dCSV("assets/c.txt",lstmNh);
        vector<float> h =  parse1dCSV("assets/h.txt",lstmNh);
        vector<float> c_ecg =  parse1dCSV("assets/c_ecg.txt",lstm_A_Nh);
        vector<float> h_ecg =  parse1dCSV("assets/h_ecg.txt",lstm_A_Nh);
        vector<float> c_w =  parse1dCSV("assets/c_w.txt",lstm_A_Nh);
        vector<float> h_w =  parse1dCSV("assets/h_w.txt",lstm_A_Nh);

        const int dim_L=510;
        const int dim_R=280;

        const int lstmWidth_A1 =  dim_L / DepthA ;
        const int lstmWidth_A2 = dim_R / DepthA ;

        //weights for alpha model
        vector<vector<float> > w0_A1 =  (parse2dCSV("assets/w0_A1.txt",dim_L, lstm_A_Nh));
        transposeMatrix(w0_A1);


        vector<vector<float> > w1_A1 =  (parse2dCSV("assets/w1_A1.txt",dim_L, lstm_A_Nh));
        transposeMatrix(w1_A1);

        vector<vector<float> > w2_A1 =  (parse2dCSV("assets/w2_A1.txt",dim_L, lstm_A_Nh));
        transposeMatrix(w2_A1);

        vector<vector<float> > w3_A1 =  (parse2dCSV("assets/w3_A1.txt",dim_L, lstm_A_Nh));
        transposeMatrix(w3_A1);

        vector<vector<float> > w0_A2 =  (parse2dCSV("assets/w0_A2.txt",dim_R, lstm_A_Nh));
        transposeMatrix(w0_A2);

        vector<vector<float> > w1_A2 =  (parse2dCSV("assets/w1_A1.txt",dim_R, lstm_A_Nh));
        transposeMatrix(w1_A2);

        vector<vector<float> > w2_A2 =  (parse2dCSV("assets/w2_A2.txt",dim_R, lstm_A_Nh));
        transposeMatrix(w2_A2);

        vector<vector<float> > w3_A2 =  (parse2dCSV("assets/w3_A2.txt",dim_R, lstm_A_Nh));
        transposeMatrix(w3_A2);


        vector<vector<float> > u0_A =  (parse2dCSV("assets/u0_A.txt",lstm_A_Nh, lstm_A_Nh));
        transposeMatrix(u0_A);

        vector<vector<float> > u1_A =  (parse2dCSV("assets/u1_A.txt",lstm_A_Nh, lstm_A_Nh));
        transposeMatrix(u1_A);

        vector<vector<float> > u2_A =  (parse2dCSV("assets/u2_A.txt",lstm_A_Nh, lstm_A_Nh));
        transposeMatrix(u2_A);

        vector<vector<float> > u3_A =  (parse2dCSV("assets/u3_A.txt",lstm_A_Nh, lstm_A_Nh));
        transposeMatrix(u3_A);
        //offsets
        vector<float> b0_A =  parse1dCSV("assets/b0_A.txt",lstm_A_Nh);
        vector<float> b1_A =  parse1dCSV("assets/b1_A.txt",lstm_A_Nh);
        vector<float> b2_A =  parse1dCSV("assets/b2_A.txt",lstm_A_Nh);
        vector<float> b3_A =  parse1dCSV("assets/b3_A.txt",lstm_A_Nh);

        //fully connected
        vector<vector<float> > fullyConnected =  (parse2dCSV("assets/fullyconnected.txt", lstmNh ,7));
        transposeMatrix(fullyConnected);
        vector<vector<float> > fullyConnected_A =  (parse2dCSV("assets/fullyconnected_A.txt", 2*lstm_A_Nh ,7));
        transposeMatrix(fullyConnected_A);
        //first layer of blend
        vector<vector<float> > fullyConnected_Blend_0 =  (parse2dCSV("assets/fullyconnected_Blend_0.txt",14,80));
        transposeMatrix(fullyConnected_Blend_0);
        //second layer of blend
        vector<vector<float> > fullyConnected_Blend_1 =  (parse2dCSV("assets/fullyconnected_Blend_1.txt",80,10));
        transposeMatrix(fullyConnected_Blend_1);
        //last layer of blend
        vector<vector<float> > fullyConnected_Blend_2 =  (parse2dCSV("assets/fullyconnected_Blend_2.txt",10,7));
        transposeMatrix(fullyConnected_Blend_2);

        vector<float> fullyConnectedB =  parse1dCSV("assets/fullyconnectedB.txt",7);

        vector<float> fullyConnectedB_A =  parse1dCSV("assets/fullyconnectedB_A.txt",7);

        vector<float> fullyConnectedB_Blend_0 =  parse1dCSV("assets/fullyconnectedB_Blend_0.txt",80);

        vector<float> fullyConnectedB_Blend_1 =  parse1dCSV("assets/fullyconnectedB_Blend_1.txt",10);

        vector<float> fullyConnectedB_Blend_2 =  parse1dCSV("assets/fullyconnectedB_Blend_2.txt",7);

        //lstm input are interpreted from ecg input which is raw input of app

        vector<float> x_ecg =  parse1dCSV("assets/x_ecg.txt",500);
        vector<float> firstRawInput = vectorcopy(x_ecg,0,250);
        vector<float> firstFeature =  parse1dCSV("assets/firstFeature.txt",4);
        vector<float> secondRawInput = vectorcopy(x_ecg,250,500);
        vector<float> secondFeature =  parse1dCSV("assets/secondFeature.txt",4);
        vector<float> xr = firstFeature;
        xr.insert(xr.end(),secondFeature.begin(), secondFeature.end());

        vector<float> highPassFilter =  parse1dCSV("assets/high_pass_filter.txt",4);
        vector<float> lowPassFilter =  parse1dCSV("assets/low_pass_filter.txt",4);

        vector<vector<float> > pca =  (parse2dCSV("assets/pca.txt", pcaInputDim_B, pcaOutputDim_B));//528,400 in this case
        transposeMatrix(pca);

        clock_t start, finish;
    for (int index = 0; index < 9; index++) { //perform everything 10 times to get a reasonable time by taking median


        //Since in each run some parameters get changed, we re-initialize those parameters in beginning  of each run to have natural conditions
        c =  parse1dCSV("assets/c.txt",lstmNh);
        h =  parse1dCSV("assets/h.txt",lstmNh);
        c_ecg =  parse1dCSV("assets/c_ecg.txt",lstm_A_Nh);
        h_ecg =  parse1dCSV("assets/h_ecg.txt",lstm_A_Nh);
        c_w =  parse1dCSV("assets/c_w.txt",lstm_A_Nh);
        h_w =  parse1dCSV("assets/h_w.txt",lstm_A_Nh);
        //lstm input are interpreted from ecg input which is raw input of app
        x_ecg =  parse1dCSV("assets/x_ecg.txt",500);

        firstRawInput = vectorcopy(x_ecg,0,250);
        firstFeature =  parse1dCSV("assets/firstFeature.txt",4);

        secondRawInput = vectorcopy(x_ecg,250,500);
        secondFeature =  parse1dCSV("assets/secondFeature.txt",4);
        xr = firstFeature;
        xr.insert(xr.end(),secondFeature.begin(), secondFeature.end());

        highPassFilter =  parse1dCSV("assets/high_pass_filter.txt",4);
        lowPassFilter =  parse1dCSV("assets/low_pass_filter.txt",4);

        pca =  (parse2dCSV("assets/pca.txt", pcaInputDim_B, pcaOutputDim_B));//528,400 in this case
        transposeMatrix(pca);

            /*
             ********************************************************************
             *****************************   wavelet start  *********************
             ********************************************************************
             */

            start=clock();
            vector<float> downsmp = downSample(firstRawInput, waveletDownSample_B);
            vector<float> wavyInput1 = wavelet(waveletOmit_B,
                    downsmp,
                    highPassFilter,
                    lowPassFilter);

            vector<float> downsmp2 = downSample(secondRawInput, waveletDownSample_B);

            vector<float> wavyInput2 = wavelet(waveletOmit_B,
                    downsmp2,
                    highPassFilter,
                    lowPassFilter);
            vector<float> downsmp1raw = downSample(firstRawInput, rawDownSample_B);
            vector<float> downsmp2raw = downSample(secondRawInput, rawDownSample_B);

            vector<float> x1 = appender(
                    downsmp1raw,
                    wavyInput1,
                    firstFeature,
                    downsmp2raw,
                    wavyInput2,
                    secondFeature);

            vector<float> x_w;
            x_w.reserve(wavyInput1.size()+wavyInput2.size());
            x_w.insert( x_w.end(), wavyInput1.begin(), wavyInput1.end() );
            x_w.insert( x_w.end(), wavyInput2.begin(), wavyInput2.end() );
            finish=clock();
            float waveletTotalTime= float(finish - start) / ((CLOCKS_PER_SEC)/1000);
            allWaveletTime[index] = waveletTotalTime;
            /*
             ********************************************************************
             *****************************   pca start  *************************
             ********************************************************************
             *//////

            start=clock();

            vector<float> betha_input=newCross(x1, pca);

            finish=clock();
            float pcatime = float(finish - start) / ((CLOCKS_PER_SEC)/1000);
            allPcaTime[index] = pcatime;

            /*
             ********************************************************************
             *****************************   Lstm start  ************************
             ********************************************************************
             */

            /*
                Model Betha
             */

            start=clock();
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

            vector<float> finalbetha = sum2Vector(newCross(h, fullyConnected), fullyConnectedB);

            finish=clock();
            lstmtime = float(finish - start) / ((CLOCKS_PER_SEC)/1000);
            allLstmTime[index] = lstmtime;
            //end of model betha

            /*
                Model alpha


             */

            vector<float> ttadd{0,0};
            vector<float> xr_ecg=concatenate(xr,x_ecg);
            xr_ecg = concatenate(xr_ecg,ttadd);
            vector<float> xr_w=concatenate(xr,x_w);
            xr_w = concatenate(xr_w,ttadd);

            start=clock();
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

            vector<float> h_final=concatenate(h_ecg,h_w);

            vector<float> finalalpha=sum2Vector(newCross(h_final, fullyConnected_A), fullyConnectedB_A);

            finish=clock();
            lstmtime = float(finish - start) / ((CLOCKS_PER_SEC)/1000);
            allLstm_A_Time[index] = lstmtime;
            //end of model alpha

            vector<float> finalVector= concatenate(finalalpha,finalbetha);

            vector<float> blend0=sum2Vector(newCross(finalVector,fullyConnected_Blend_0),fullyConnectedB_Blend_0); // to 80 output layer

            vector<float> blend=sum2Vector(newCross(blend0,fullyConnected_Blend_1),fullyConnectedB_Blend_1);  // to 10 output layer

            vector<float> result=sum2Vector(newCross(blend,fullyConnected_Blend_2),fullyConnectedB_Blend_2);
            }

            sort(allWaveletTime, allWaveletTime+ (sizeof(allWaveletTime)/sizeof(allWaveletTime[0])));
            sort(allPcaTime,allPcaTime+ (sizeof(allPcaTime)/sizeof(allPcaTime[0])));
            sort(allLstmTime, allLstmTime+ (sizeof(allLstmTime)/sizeof(allLstmTime[0])));
            sort(allLstm_A_Time, allLstm_A_Time+ (sizeof(allLstm_A_Time)/sizeof(allLstm_A_Time[0])));

            float totalTime = allWaveletTime[4] + allPcaTime[4] + allLstmTime[4] + allLstm_A_Time[4] + allblend_Time[4];
            cout<<"finished successfully!"<<endl;
            cout<<"algorithm total execution time is: "<<totalTime<<endl<<"per task is: "<<allWaveletTime[4]<<","<<allPcaTime[4]<<","<<allLstmTime[4]<<","<<allLstm_A_Time[4]<<","<<allblend_Time[4];

    return 0;
}




vector<float> convolution1D(vector<float> inputSignal, vector<float> kernel) {

        kernel = reverser(kernel);

        float inputLength = inputSignal.size();
        float mirrorPaddingSizeForBeginningOfArray = (float) kernel.size() - 2;
        float mirrorPaddingSizeForEndOfArray = (float) kernel.size() - 2;

        if((int)inputLength % 2 != 0){
            mirrorPaddingSizeForEndOfArray++;
        }

        float outputLength = (float) ceil(((inputLength - kernel.size() + mirrorPaddingSizeForBeginningOfArray + mirrorPaddingSizeForEndOfArray) / 2) + 1);

        inputSignal = makeMirrorPadding(inputSignal, (int) mirrorPaddingSizeForBeginningOfArray, (int) mirrorPaddingSizeForEndOfArray);
        vector<float> output((int)outputLength,0);

        int index = 0;
        for (int i = 0; i < outputLength * 2; i += 2) {
            output[index] = multiplier(inputSignal, kernel, i);
            index++;
        }

        return output;
    }

vector<float> makeMirrorPadding(vector<float> &inputSignal, int mirrorPaddingSizeForBeginningOfArray, int mirrorPaddingSizeForEndOfArray) {


        int inputLength = inputSignal.size();
        vector<float> outPutVector(inputLength + mirrorPaddingSizeForBeginningOfArray + mirrorPaddingSizeForEndOfArray,0);

        int counter = 0;
        for (int index = 0; index < mirrorPaddingSizeForBeginningOfArray; index++) {
            outPutVector[index] = inputSignal[mirrorPaddingSizeForBeginningOfArray - 1 - index];
        }

        for (int index = mirrorPaddingSizeForBeginningOfArray; index < inputLength + mirrorPaddingSizeForBeginningOfArray; index++) {
            outPutVector[index] = inputSignal[counter];
            counter++;
        }

        counter = 1;
        for (int index = inputLength + mirrorPaddingSizeForBeginningOfArray; index < (inputLength + mirrorPaddingSizeForBeginningOfArray + mirrorPaddingSizeForEndOfArray); index++) {
            outPutVector[index] = inputSignal[inputLength - counter];
            counter++;
        }

        return outPutVector;
}
//defining some simple functions for vector manipulation

vector<float> reverser(vector<float> &input){

        vector<float> output(input.size(),0);

        int kernelSize = input.size();
        for(int index = 0; index < kernelSize; index++){
            output[index] = input[kernelSize - 1 - index];
        }
        return output;
}


float multiplier(vector<float> &inputSignal, vector<float> &kernel, int startIndexOfInput) {

        float sum = 0;
        int kernelIndex = 0;
        for (int index = startIndexOfInput; index < kernel.size() + startIndexOfInput; index++, kernelIndex++) {
            sum += inputSignal[index] * kernel[kernelIndex];
        }

        return sum;
    }


vector<float> wavelet(int omit, vector<float> &inputSignal, vector<float> &highPassFilter, vector<float> &lowPassFilter) {

        //Convolution convolution = new Convolution();

        vector<float> convolutionWithHighPass1 = convolution1D(inputSignal, highPassFilter);
        vector<float> convolutionWithLowPass1 = convolution1D(inputSignal, lowPassFilter);

        vector<float> convolutionWithHighPass2 = convolution1D(convolutionWithLowPass1, highPassFilter);
        vector<float> convolutionWithLowPass2 = convolution1D(convolutionWithLowPass1, lowPassFilter);

        vector<float> convolutionWithHighPass3 = convolution1D(convolutionWithLowPass2, highPassFilter);
        vector<float> convolutionWithLowPass3 = convolution1D(convolutionWithLowPass2, lowPassFilter);

        vector<float> convolutionWithHighPass4 = convolution1D(convolutionWithLowPass3, highPassFilter);
        vector<float> convolutionWithLowPass4 = convolution1D(convolutionWithLowPass3, lowPassFilter);

        if(omit == 1){
            vector<float> temp(1,0);
            convolutionWithHighPass1.swap(temp);
        }
        if(omit == 2){
            vector<float> temp(1,0);
            convolutionWithHighPass1.swap(temp);
            convolutionWithHighPass2.swap(temp);
        }

        int length = convolutionWithHighPass1.size() + convolutionWithHighPass2.size() + convolutionWithHighPass3.size() + convolutionWithHighPass4.size() + convolutionWithLowPass4.size();
        vector<float> finalResult(length,0);

        int index = 0;
        for (auto &temp : convolutionWithLowPass4) {
            finalResult[index] = temp;
            index++;
        }

        for (auto &temp : convolutionWithHighPass4) {
            finalResult[index] = temp;
            index++;
        }

        for (auto &temp : convolutionWithHighPass3) {
            finalResult[index] = temp;
            index++;
        }

        for (auto &temp : convolutionWithHighPass2) {
            finalResult[index] = temp;
            index++;
        }

        for (auto &temp : convolutionWithHighPass1) {
            finalResult[index] = temp;
            index++;
        }

        return finalResult;
}


int pcaInputCalculator(int rawDownSample, int waveletDownSample, int waveletOmit) {

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

vector<float> dot(vector<float> in1,const vector<float> &in2) {

    if (in2.size() == 1) {
        for (int i = 0; i < in1.size(); i++) {
            in1[i] = in1[i] * in2[0];
        }
    } else {
        for (int i = 0; i < in1.size(); i++) {
            in1[i] = in1[i] * in2[i];
        }
    }

    return in1;
}


vector<float> sum3vector(vector<float> in1,const vector<float> &in2,const vector<float> &in3) {

    for (int i = 0; i < in1.size(); i++) {
        in1[i] = in1[i] + in2[i] + in3[i];
    }

    return in1;
}

vector<float> sum2Vector(vector<float> in1,const vector<float> &in2) {

    for (int i = 0; i < in1.size(); i++) {
        in1[i] = in1[i] + in2[i];
    }

    return in1;
}



vector<float> tanHEval(vector<float> in) {

    for (int i = 0; i < in.size(); i++) {
        in[i] = (float)tanh(in[i]);
    }

    return in;
}

vector<float> sigmoid(vector<float> in) {
    for (int i = 0; i < in.size(); i++) {
        in[i] =  (float)1 / (1 + exp(-1 * in[i]));
    }
    return in;
}

void transposeMatrix(vector<vector<float> > &b)
{
    if (b.size() == 0)
        return;
    vector<float> row(b.size(),0);
    vector<vector<float> > trans_vec(b[0].size(), row);

    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < b[i].size(); j++)
        {
            trans_vec[j].push_back(b[i][j]);
        }
    }

    b.swap(trans_vec);
}
vector<float> newCrossInRange(const vector<float> &in1,const vector<vector<float> > &in2, int j1, int j2) {

    int n = in2.size();
    vector<float> res(n,0);

    for (int i = 0; i < n; i++) {
        for (int j = j1; j < j2; j++) {
            res[i] += in2[i][j] * in1[j];
        }
    }
    return res;
}

vector<float> newCross(const vector<float> &in1,const vector<vector<float>> &in2) {

    int n = in2.size();//600
    int m = in2[0].size();//1026

    vector<float> res(n,0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res[i] += in2[i][j] * in1[j];
        }
    }
    return res;
}

vector<float> appender(vector<float> &x1, vector<float> &x2,vector<float> &x3, vector<float> &x4,vector<float> &x5, vector<float> &x6) {
    vector<float> output (x1.size() + x2.size()+ x3.size() + x4.size() + x5.size() + x6.size(),0);

    int index = 0;
    for (auto &temp : x1) {
        output[index] = temp;
        index++;
    }
    for (auto &temp : x2) {
        output[index] = temp;
        index++;
    }
    for (auto &temp : x3) {
        output[index] = temp;
        index++;
    }
    for (auto &temp : x4) {
        output[index] = temp;
        index++;
    }
    for (auto &temp : x5) {
        output[index] = temp;
        index++;
    }
    for (auto &temp : x6) {
        output[index] = temp;
        index++;
    }
    return output;
}


vector<float> concatenate(vector<float> &x1, vector<float> &x2) {

    vector<float> output;
    output.reserve(x1.size()+x2.size());
    output.insert( output.end(), x1.begin(), x1.end() );
    output.insert( output.end(), x2.begin(), x2.end() );

    return output;
}


vector<float> downSample(vector<float> &x,int rate){
    if (rate == 1)
        return x;
    else {
        vector<float> output(x.size()/2,0);
        int index = 0;
        for (int i = 0; i < x.size(); i += 2, index++) {
            output[index] = x[i];
        }
        return output;
    }

}

vector<float> vectorcopy(vector<float> &a, int b0 , int b1){

    vector<float>::const_iterator first = a.begin() + b0;
    vector<float>::const_iterator last = a.begin() + b1;
    vector<float> b(first, last);
    return b;

}

//To read matrices from txt (csv styled) files

vector<vector<float> > parse2dCSV(string filename,int m,int n)
{
    ifstream  data(filename.c_str());
    string line;
    vector<vector<float> > parsedCsv;
    while(getline(data,line))
    {
        stringstream lineStream(line.c_str());
        string cell;
        vector<float> parsedRow;
        while(getline(lineStream,cell,','))
        {
            float dcell = stof(cell);
            parsedRow.push_back(dcell);
        }

        parsedCsv.push_back(parsedRow);
    }
    return parsedCsv;

}

//To read vectors from txt (csv styled) files

vector<float> parse1dCSV(string filename,int n)
{
    ifstream  data(filename);
    string line;
    getline(data,line);

    stringstream lineStream(line);
    string cell;
    vector<float> parsedRow;
    while(getline(lineStream,cell,','))
    {
        float dcell = stof(cell);
        parsedRow.push_back(dcell);
    }


    return parsedRow;

}
