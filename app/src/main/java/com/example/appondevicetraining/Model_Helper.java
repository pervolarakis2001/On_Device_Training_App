package com.example.appondevicetraining;

import android.app.AlertDialog;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.logging.Handler;
import java.util.Random;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.lite.DataType;

public class Model_Helper {
    private int numThreads = 2;
    private Context context;

    private Interpreter interpreter = null;
    private List<TrainingImage> TrainingImages = new ArrayList<>();


    private int targetWidth = 0;
    private int targetHeight = 0;


    public Model_Helper(Context context) {
        this.context = context;
        if (setupModelPersonalization()) {
            int[] shape = interpreter.getInputTensor(0).shape();

            this.targetWidth = shape[2];
            this.targetHeight = shape[1];
            Log.d("Model_Helper", "Model personalization successfulu  initialized");


        } else {
            Log.d("Model_Helper", "Model personalization failed to initialize");

        }
    }

    // load model and check if is loaded correctly
    public boolean setupModelPersonalization() {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(numThreads);
        AssetManager assetManager = context.getAssets();
        try {
            ByteBuffer model = loadModelFile(assetManager, "model/model.tflite");
            interpreter = new Interpreter(model, options);
            return true;
        } catch (IOException e) {
            Log.e("Model", "TFLite failed to load model with error: " + e.getMessage());
            return false;
        }
    }

    private static ByteBuffer loadModelFile(AssetManager assetManager, String filename) throws IOException {
        InputStream inputStream = null;
        try {
            inputStream = assetManager.open(filename);
            int fileSize = inputStream.available();
            if (fileSize <= 0) {
                throw new IOException("Model file is empty: " + filename);
            }
            ByteBuffer buffer = ByteBuffer.allocateDirect(fileSize);
            byte[] byteBuffer = new byte[fileSize];
            int bytesRead;
            while ((bytesRead = inputStream.read(byteBuffer)) != -1) {
                buffer.put(byteBuffer, 0, bytesRead);
            }
            buffer.rewind();
            return buffer;
        } finally {
            if (inputStream != null) {
                inputStream.close();
            }
        }
    }


    public void Close() {
        interpreter = null;
    }


    // preprocess training images and convert them to TensorImage for classification
    public TensorImage PreprocessImages(Bitmap image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int cropSize = Math.min(height, width);
        ImageProcessor.Builder imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(
                        targetHeight,
                        targetWidth,
                        ResizeOp.ResizeMethod.BILINEAR
                ))
                .add(new NormalizeOp(0f, 255f));
        ImageProcessor imageProcessorBuilder = imageProcessor.build();
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(image);
        return imageProcessorBuilder.process(tensorImage);
    }

    // One-Hot encode labels
    public List<Float> encoding(int n, int numClasses) {
        List<Float> oneHotEncoded = new ArrayList<>(numClasses);
        for (int i = 0; i < numClasses; i++) {
            oneHotEncoded.add(0.0f); // Initialize all elements with 0.0f
        }
        oneHotEncoded.set(n, 1.0f); // Set the nth element to 1.0f
        return oneHotEncoded;
    }

    public void Training(List<TrainingSample> ds_train_m, HashMap<List<Float>, List<TensorImage>> ds_train_n) {
        int num_Samples = 5;
        int training_steps = 611;
        int NUM_EPOCHS = 1;
        int IMG_SIZE = 160;
        Random random = new Random();
        if (interpreter == null) {
            setupModelPersonalization();
        }

        if (ds_train_m.size() < num_Samples) {
            throw new RuntimeException(
                    String.format(
                            "Too few samples to start training: need %d, got %d",
                            num_Samples, ds_train_m.size()
                    )
            );
        }

        int c = 0;
        int counter = 0;
        List<Float> losses = new ArrayList<>();
        Map<TrainingSample, Integer> m_dict = new HashMap<>();

        for (Map.Entry<List<Float>, List<TensorImage>> entry : ds_train_n.entrySet()) {
            List<Float> key = entry.getKey();
            List<TensorImage> value = entry.getValue();

        }

        for (TrainingSample m_sample : ds_train_m) {
            List<TensorImage> X_train = new ArrayList<>();
            List<List<Float>> Y_train = new ArrayList<>();
            X_train.add(m_sample.getImage());
            Y_train.add(m_sample.getLabel());

            if (!m_dict.containsKey(m_sample.getImage())) {
                // choose randomly 4 labels from n classes
                List<List<Float>> n_classes = new ArrayList<>(ds_train_n.keySet());

                for (int i = n_classes.size() - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    List<Float> temp = n_classes.get(i);
                    n_classes.set(i, n_classes.get(j));
                    n_classes.set(j, temp);
                }
                List<List<Float>> selectedClasses = n_classes.subList(0, Math.min(n_classes.size(), 4));
                HashMap<List<Float>, List<TensorImage>> ds_n = new HashMap<>();
                for (List<Float> key : selectedClasses) {
                    List<TensorImage> value = ds_train_n.get(key);
                    ds_n.put(key, value);
                }


                //  concatenate m and n classes
                Set<List<Float>> n_labels = ds_n.keySet();

                for (List<Float> label : n_labels) {
                    List<TensorImage> x_n = ds_n.get(label);
                    X_train.addAll(x_n);
                    for (int i = 0; i < 5; i++) {
                        Y_train.add(label);
                    }
                }


                // Convert X_train, Y_train to appropriate types
                int numImages = X_train.size();
                int totalDataSize = 1 * IMG_SIZE * IMG_SIZE * 3 * Float.SIZE / Byte.SIZE;
                ByteBuffer inputBuffer = ByteBuffer.allocateDirect(totalDataSize);
                inputBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

                int labelBufferSize = 1 * 10 * Float.SIZE / Byte.SIZE;
                ByteBuffer labelBuffer = ByteBuffer.allocateDirect(labelBufferSize);
                labelBuffer.order(ByteOrder.nativeOrder());

                for (int i = 0; i < numImages; i++) {
                    labelBuffer.clear();
                    inputBuffer.clear();

                    ByteBuffer imageData = X_train.get(i).getBuffer();
                    imageData.rewind();
                    inputBuffer.put(imageData);
                    inputBuffer.rewind();

                    List<Float> y = Y_train.get(i);
                    for (int j = 0; j < y.size(); j++) {
                        labelBuffer.putFloat(y.get(j));
                    }
                    labelBuffer.rewind();

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", inputBuffer);
                    inputs.put("y", labelBuffer);
                    Map<String, Object> outputs = new HashMap<>();

                    FloatBuffer loss = FloatBuffer.allocate(1);
                    outputs.put("loss", loss);

                    interpreter.runSignature(inputs, outputs, "train");

                    // Record the last loss.
                    if (i == numImages - 1) losses.add(loss.get(0));
            }
                c++;
                System.out.println(
                        "Finished iteration " + counter + ", current loss: " + losses.get(losses.size()-1));

            int k = random.nextInt(6) + 5;
            m_dict.put(new TrainingSample(m_sample.getImage(), m_sample.getLabel()), counter + k);


        }

        for (Map.Entry<TrainingSample, Integer> entry : m_dict.entrySet()) {
            TrainingSample m_data = entry.getKey();
            int next_i = entry.getValue();

            List<TensorImage> new_X_train = new ArrayList<>();
            List<List<Float>> new_Y_train = new ArrayList<>();
            new_X_train.add(m_data.getImage());
            new_Y_train.add(m_sample.getLabel());

            if (next_i == counter) {
                Log.d("checkTraining" , "it runs");
                int k = random.nextInt(6) + 5;
                m_dict.put(m_data, m_dict.getOrDefault(m_data, 0) + k);

                // choose randomly 4 labels from n classes
                List<List<Float>> n_classes = new ArrayList<>(ds_train_n.keySet());

                for (int i = n_classes.size() - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    List<Float> temp = n_classes.get(i);
                    n_classes.set(i, n_classes.get(j));
                    n_classes.set(j, temp);
                }
                List<List<Float>> selectedClasses = n_classes.subList(0, Math.min(n_classes.size(), 4));
                HashMap<List<Float>, List<TensorImage>> ds_n = new HashMap<>();
                for (List<Float> key : selectedClasses) {
                    List<TensorImage> value = ds_train_n.get(key);
                    ds_n.put(key, value);
                }

                //  concatenate m and n classes
                Set<List<Float>> n_labels = ds_n.keySet();
                for (List<Float> label : n_labels) {
                    List<TensorImage> x_n = ds_n.get(label);
                    new_X_train.addAll(x_n);
                    for (int i = 0; i < 5; i++) {
                        new_Y_train.add(label);
                    }
                }
                // Convert X_train, Y_train to appropriate types

                int newnumImages = new_X_train.size();
                int totalDataSize = 1 * IMG_SIZE * IMG_SIZE * 3 * Float.SIZE / Byte.SIZE;
                ByteBuffer newinputBuffer = ByteBuffer.allocateDirect(totalDataSize);
                newinputBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

                int labelBufferSize = 1 * 10 * Float.SIZE / Byte.SIZE;
                ByteBuffer newlabelBuffer = ByteBuffer.allocateDirect(labelBufferSize);
                newlabelBuffer.order(ByteOrder.nativeOrder());

                for (int i = 0; i < newnumImages; i++) {
                    newlabelBuffer.clear();
                    newinputBuffer.clear();

                    ByteBuffer imageData = new_X_train.get(i).getBuffer();
                    imageData.rewind();
                    newinputBuffer.put(imageData);
                    newinputBuffer.rewind();

                    List<Float> newy = new_Y_train.get(i);
                    for (int j = 0; j < newy.size(); j++) {
                        newlabelBuffer.putFloat(newy.get(j));
                    }
                    newlabelBuffer.rewind();

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", newinputBuffer);
                    inputs.put("y", newlabelBuffer);


                    Map<String, Object> outputs = new HashMap<>();
                    FloatBuffer loss = FloatBuffer.allocate(1);
                    outputs.put("loss", loss);

                    interpreter.runSignature(inputs, outputs, "train");


                    if (i == newnumImages - 1) losses.add(loss.get(0));
                }
                c++;
                System.out.println(
                        "Finished iteration " + counter + ", current loss: " + losses.get(losses.size()-1));
            }

        }

        counter++;
    }


}

    public void Classify(TensorImage image){
        int totalDataSize = 1 * 160 * 160 * 3 * Float.SIZE / Byte.SIZE;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(totalDataSize);
        inputBuffer.order(ByteOrder.nativeOrder());


        ByteBuffer imageData = image.getBuffer();
        imageData.rewind();
        inputBuffer.put(imageData);
        inputBuffer.rewind();

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("x", inputBuffer );


        Map<String, Object> outputs = new HashMap<>();
        FloatBuffer probabilities = FloatBuffer.allocate(10);
        probabilities.rewind();
        outputs.put("output", probabilities);

        interpreter.runSignature(inputs, outputs, "infer");
        int maxIndex = -1;
        float maxValue = Float.MIN_VALUE; // Initialize maxValue with the smallest possible float value
        int i =0 ;
        probabilities.rewind();
        while (probabilities.hasRemaining())  {
            float value = probabilities.get();
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
            i++;
        }

        System.out.println("Predicted category " + maxIndex);
    }

}




