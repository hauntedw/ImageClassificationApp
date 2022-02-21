package com.example.imageclassificationapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    Uri selectedImage;
    ImageView imageView;
    Bitmap bitmap = null;
    Module module = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //loading pytorch model
        try {
            module = LiteModuleLoader.load(assetFilePath(this, "model.pt"));
        } catch (IOException e) {
            Log.e("PytorchTest", "Error reading assets", e);
            finish();
        }

        //requesting permissions to use phone storage
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        //creating click listener for gallery button
        Button gallery = findViewById(R.id.gallery);
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 3);
            }
        });

        //creating click listener for identify button
        Button identify = findViewById(R.id.identify);
        identify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // creating input Tensor
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                        //normalizing image with mean and std
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                //generating output
                final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

                final float[] scores = outputTensor.getDataAsFloatArray();

                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;
                //finding most likely match
                for (int i= 0; i < scores.length; i++){
                    if (scores[i] > maxScore){
                        maxScore = scores[i];
                        maxScoreIdx = i;
                    }
                }

                //setting classname to most likely score
                String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

                //setting textview to match classname
                TextView textView = findViewById(R.id.textView);
                textView.setText(className);
            }
        });
    }

    //onresult occurs when user selects an image
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == RESULT_OK && data != null) {
            //selectedImage variable is set and imageView is updated to selectedImage
            selectedImage = data.getData();
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageURI(selectedImage);
            //generate a bitmap of the selectedImage
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), selectedImage);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    //setting assetFilePath name to load model
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}