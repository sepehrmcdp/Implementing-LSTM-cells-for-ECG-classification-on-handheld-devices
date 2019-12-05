package DPHC.DPHC2.readmatfile;

import android.content.Context;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class readmatfiles {


    private Context context;

    public readmatfiles(Context context) {
        this.context = context;
    }


    public float[][] get2DFloats(String fileName,int m , int n)  throws IOException{
        float[][] datas = new float[m][n];

            BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(fileName)));

            String line = null;


            for (int j = 0; j < m; j++) {
                line = br.readLine();
                String[] values = line.split(",");

                // looping over String values
                for (int i = 0; i < n; i++) {
                    // trying to parse String value as float

                    // worked, assigning to respective float[] array position
                    datas[j][i] = Float.parseFloat(values[i]);

                }


            }

            br.close();


        return datas;
    }
    public float[] get1DFloats(String fileName,int m)  throws Exception{


        BufferedReader br = new BufferedReader(new InputStreamReader(context.getAssets().open(fileName)));

        String line = null;
        float[] datas = new float[m];

            line = br.readLine();
            String[] values = line.split(",");

            // looping over String values
            for (int i = 0; i < m; i++) {
                // trying to parse String value as float

                // worked, assigning to respective float[] array position
                datas[i] = Float.parseFloat(values[i]);

            }




        br.close();
        return datas;


    }
}
