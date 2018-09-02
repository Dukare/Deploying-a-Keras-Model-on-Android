package com.example.android_app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.Toast;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;

    private static final String MODEL_FILE = "file:///android_asset/frozen_cifar10.pb";
    private static final String INPUT_NODE = "conv2d_1_input";
    private static final String[] OUTPUT_NODES = {"dense_3/Softmax"};
    private static final String OUTPUT_NODE = "dense_3/Softmax";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        GridView gridview = (GridView) findViewById(R.id.gridView);
        gridview.setAdapter(new ImageAdapter(this));

        gridview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            public void onItemClick(AdapterView<?> parent, View v,
                                    int position, long id) {

                final int inputSize = 32;
                final int destWidth = 32;
                final int destHeight = 32;

                final String fileName = "sample_" + (position + 1) + ".png";

                InputStream file = null;
                try{
                    file = getAssets().open(fileName);
                }catch (Exception e){
                    e.printStackTrace();
                    return;
                }

                Bitmap bitmap = BitmapFactory.decodeStream(file);

                Bitmap bitmap_scaled = Bitmap.createScaledBitmap(bitmap, destWidth, destHeight, false);

                int[] intValues = new int[inputSize * inputSize]; // array to copy values from Bitmap image
                float[] floatValues = new float[inputSize * inputSize * 3]; // float array to store image data

                bitmap_scaled.getPixels(intValues, 0, bitmap_scaled.getWidth(), 0, 0, bitmap_scaled.getWidth(), bitmap_scaled.getHeight());

                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                    floatValues[i * 3 + 0] = ((val >> 16) & 0xFF) / 255;
                    floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255;
                    floatValues[i * 3 + 2] = (val & 0xFF) / 255;
                }

                inferenceInterface.feed(INPUT_NODE, floatValues, 1L, 32L, 32L, 3L);
                inferenceInterface.run(OUTPUT_NODES);

                float[] result = new float[10];
                inferenceInterface.fetch(OUTPUT_NODE, result);

                String[] classes = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog ", "horse", "ship", "truck"};

                String resultStr = "";
                for (int i = 0; i < 10; ++i){
                    if (result[i] * 100 > 1L){
                        resultStr += (result[i] * 100) + " " + classes[i] + " | ";
                    }
                }

                Toast.makeText(MainActivity.this, "position " + position + ", result " + resultStr,
                        Toast.LENGTH_LONG).show();
            }
        });
    }
}

class ImageAdapter extends BaseAdapter {
    private Context mContext;

    public ImageAdapter(Context c) {
        mContext = c;
    }

    public int getCount() {
        return mThumbIds.length;
    }

    public Object getItem(int position) {
        return null;
    }

    public long getItemId(int position) {
        return 0;
    }

    // create a new ImageView for each item referenced by the Adapter
    public View getView(int position, View convertView, ViewGroup parent) {
        ImageView imageView;
        if (convertView == null) {
            // if it's not recycled, initialize some attributes
            imageView = new ImageView(mContext);
            imageView.setLayoutParams(new ViewGroup.LayoutParams(150, 150));
            imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
            imageView.setPadding(8, 8, 8, 8);
        } else {
            imageView = (ImageView) convertView;
        }

        imageView.setImageResource(mThumbIds[position]);
        return imageView;
    }

    // references to our images
    private Integer[] mThumbIds = {
            R.drawable.sample_1,
            R.drawable.sample_2,
            R.drawable.sample_3,
            R.drawable.sample_4,
            R.drawable.sample_5,
            R.drawable.sample_6
    };
}