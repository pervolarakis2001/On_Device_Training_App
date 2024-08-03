package com.example.appondevicetraining;

import static java.lang.Math.abs;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.ProgressBar;

import androidx.fragment.app.Fragment;

import com.google.android.gms.tflite.gpu.GpuDelegate;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Model_Helper extends Fragment {
    private int numThreads = 2;
    private Context context;
    private ProgressBar progressBar;
    private Interpreter interpreter = null;
    private List<TrainingImage> TrainingImages = new ArrayList<>();


    private int targetWidth = 0;
    private int targetHeight = 0;

    public interface OnTrainingCompleteListener {
        void onTrainingComplete();
    }

    private OnTrainingCompleteListener onTrainingCompleteListener;

    public void setOnTrainingCompleteListener(OnTrainingCompleteListener listener) {
        this.onTrainingCompleteListener = listener;
    }

    public interface OnProgressUpdateListener {
        void onProgressUpdate(int progress);
        void onTrainingComplete();
    }
    public Model_Helper(Context context) {
        this.context = context;
        if (setupModelPersonalization()) {
            int[] shape = interpreter.getInputTensor(0).shape();
            System.out.println(shape[0]);
            this.targetWidth = shape[2];
            this.targetHeight = shape[1];
            Log.d("Model_Helper", "Model personalization successfulu  initialized");


        } else {
            Log.d("Model_Helper", "Model personalization failed to initialize");

        }
    }

    // load model and check if is loaded correctly
    public boolean setupModelPersonalization() {
        //Interpreter.Options options = new Interpreter.Options();


        AssetManager assetManager = context.getAssets();
        try {
            ByteBuffer model = loadModelFile(assetManager, "model/model.tflite");
            Interpreter.Options options = new Interpreter.Options();
            options.setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY);
            options.addDelegateFactory(new GpuDelegateFactory());
            options.setNumThreads(numThreads);
            interpreter = new Interpreter(model, options);
            interpreter.allocateTensors();
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
                //.add(new ResizeWithCropOrPadOp(cropSize, cropSize))
               // .add(new ResizeOp(
               //         targetHeight,
                 //     targetWidth,
                //       ResizeOp.ResizeMethod.BILINEAR
               //))
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

    public void Training(List<TrainingSample> ds_train_m, HashMap<Integer, List<Bitmap>> train_n, String numClass,HashMap<List<Float>, List<TensorImage>> ds_test_n) {
        int num_Samples = 2;
        int training_steps = 611;
        int NUM_EPOCHS = 1;
        int IMG_SIZE = 160;
        Random random = new Random();
        if (interpreter == null) {
            setupModelPersonalization();
        }
        int[] shape = interpreter.getInputTensor(0).shape();
        System.out.println(shape[0]);
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
        List<Float> preds = new ArrayList<>();
        List<List<Float>> actual = new ArrayList<>();
        Map<TrainingSample, Integer> m_dict = new HashMap<>();

        // To save results
        List<Object> acc_n = new ArrayList<>();
        List<Object> acc_m = new ArrayList<>();
        List<Object> time = new ArrayList<>();
        List<Object> losses = new ArrayList<>();

        // choose 5 images per n class
        HashMap<List<Float>, List<TensorImage>> copy_ds_train_n =  new HashMap<>();
        for (Map.Entry<Integer, List<Bitmap>> entry : train_n.entrySet()) {
            int key = entry.getKey();
            List<TensorImage> copiedValues = new ArrayList<>();
            List<Bitmap> retrievedValues = entry.getValue();
            for(Bitmap image: retrievedValues){
                if(copiedValues.size()==5){
                    break;
                }
                TensorImage img = PreprocessImages(image);
                copiedValues.add(img);
            }
            List<Float> label = encoding(key,10);


            copy_ds_train_n.put(label, copiedValues);
        }


        HashMap<List<Float>, List<TensorImage>> ds_train_n = new HashMap<>();
        ds_train_n = copy_ds_train_n;

        for (TrainingSample m_sample : ds_train_m) {


            List<TensorImage> X_train = new ArrayList<>();
            List<List<Float>> Y_train = new ArrayList<>();

            if (!m_dict.containsKey(new TrainingSample(m_sample.getImage(), m_sample.getLabel()))) {
                // choose randomly 4 labels from n classes
                List<List<Float>> n_classes = new ArrayList<>(ds_train_n.keySet());

                for (int i = n_classes.size() - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    List<Float> temp = n_classes.get(i);
                    n_classes.set(i, n_classes.get(j));
                    n_classes.set(j, temp);
                }
                List<List<Float>> selectedClasses = n_classes.subList(0, Math.min(n_classes.size(), 4));
                HashMap<TensorImage, List<Float>> ds_n = new HashMap<>();
                for (List<Float> key : selectedClasses) {
                    List<TensorImage> value = ds_train_n.get(key);
                    int randomIndex = random.nextInt(value.size());
                    TensorImage randomImage = value.get(randomIndex);

                    ds_n.put(randomImage, key);
                }

                //  concatenate m and n classes

                int totalDataSize = 5 * IMG_SIZE * IMG_SIZE * 3;
                FloatBuffer inputBuffer = ByteBuffer.allocateDirect(totalDataSize * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();



               // int labelBufferSize = 5 * 10 * Float.SIZE / Byte.SIZE;
               // ByteBuffer labelBuffer = ByteBuffer.allocateDirect(labelBufferSize).order(ByteOrder.nativeOrder());

                // Add m_sample data


                for (Map.Entry<TensorImage, List<Float>> e : ds_n.entrySet()) {
                    TensorImage x_n = e.getKey();
                    List<Float> y_n = e.getValue();

                    X_train.add(x_n);
                    Y_train.add(y_n);

                }
                X_train.add(m_sample.getImage() );
                Y_train.add(m_sample.getLabel());
                int numImages = X_train.size();
                System.out.println(numImages);

                // Convert X_train, Y_train to appropriate types
                float[][][][] trainImages = new float[5][160][160][3];
                float[][] trainLabels = new float[5][10];

                for (int i = 0; i < numImages; i++) {
                   // labelBuffer.clear();
                   // inputBuffer.clear();

                    TensorImage imageData = X_train.get(i);
                    TensorBuffer tensorBuffer = imageData.getTensorBuffer();

                    float[] imageFloatArray = tensorBuffer.getFloatArray();
                    //inputBuffer.put(imageFloatArray);
                    int index = 0;
                    for (int row = 0; row < 160; row++) {
                        for (int col = 0; col < 160; col++) {
                            for (int channel = 0; channel < 3; channel++) {
                                trainImages[i][row][col][channel] = imageFloatArray[index++];
                            }
                        }
                    }

                    //List<Float> y = Y_train.get(i);
                    //for (float label : y) {
                    //    labelBuffer.putFloat(label);
                    //}
                    List<Float> label = Y_train.get(i);
                    for (int j = 0; j < 10; j++) {
                        trainLabels[i][j] = label.get(j);
                    }

                    // Record the last loss.
                   // if (i == numImages - 1) losses.add(loss.get(0));

                }


                //inputBuffer.rewind();
                //labelBuffer.rewind();

                Map<String, Object> inputs = new HashMap<>();
                inputs.put("x", trainImages);
                inputs.put("y", trainLabels);
                Map<String, Object> outputs = new HashMap<>();

                FloatBuffer loss = FloatBuffer.allocate(1);
                FloatBuffer prediction = FloatBuffer.allocate(5* 10 * Float.SIZE / Byte.SIZE);

                outputs.put("loss", loss);


                long startTime = System.nanoTime();
                interpreter.runSignature(inputs, outputs,"train");
                long endTime = System.nanoTime();
                long elapsedTime = endTime - startTime;
                double elapsedTimeInSeconds = elapsedTime / 1_000_000_000.0;
                time.add(elapsedTimeInSeconds);

                losses.add(loss.get(0));





                //TODO: EVALUATION ON ds_test_m AND ds_test_n
                //         accuracy = number_of_correct_predictions / total_samples
                Map<String, Object> accuracies = EvaluateModel(ds_test_n, ds_train_m);
                for (Map.Entry<String, Object> entry : accuracies.entrySet()) {
                    String key = entry.getKey();
                    Object value = entry.getValue();

                    // Add to the appropriate list based on the key
                    if (key.equals("accuracy_m")) {
                        acc_m.add(value);
                    } else if (key.equals("accuracy_n")) {
                        acc_n.add(value);
                    }
                }

                if (abs((float) acc_n.get(acc_n.size() - 1) - (float) acc_m.get(acc_m.size() - 1)) <= 0.02) {
                    System.out.println(
                            "For Test sets, the accuracies were equal in iteration" + counter + " with total training steps=" + c + ", accuracy=" + acc_m.get(acc_m.size() - 1) + ", and m_samples={}");
                }
                System.out.println(accuracies);
                c++;
                System.out.println(
                        "Finished iteration " + counter + ", current loss: " + losses.get(losses.size() - 1));


                int k = random.nextInt(6) + 5;
                m_dict.put(new TrainingSample(m_sample.getImage(), m_sample.getLabel()), counter + k);


                for (Map.Entry<TrainingSample, Integer> entry : m_dict.entrySet()) {
                    TrainingSample m_data = entry.getKey();
                    int next_i = entry.getValue();

                    List<TensorImage> new_X_train = new ArrayList<>();
                    List<List<Float>> new_Y_train = new ArrayList<>();

                    if (next_i == counter) {

                        Log.d("checkTraining", "it runs");
                        int w = random.nextInt(6) + 5;
                        m_dict.put(m_data, m_dict.getOrDefault(m_data, 0) + w);

                        // choose randomly 4 labels from n classes
                        List<List<Float>> newn_classes = new ArrayList<>(ds_train_n.keySet());

                        for (int i = newn_classes.size() - 1; i > 0; i--) {
                            int j = random.nextInt(i + 1);
                            List<Float> temp = newn_classes.get(i);
                            newn_classes.set(i, newn_classes.get(j));
                            newn_classes.set(j, temp);
                        }
                        List<List<Float>> newselectedClasses = newn_classes.subList(0, Math.min(newn_classes.size(), 4));
                        HashMap<TensorImage, List<Float>> newds_n = new HashMap<>();
                        for (List<Float> key : newselectedClasses) {
                            List<TensorImage> value = ds_train_n.get(key);
                            int randomIndex = random.nextInt(value.size());
                            TensorImage randomImage = value.get(randomIndex);

                            newds_n.put(randomImage, key);
                        }

                        //  concatenate m and n classes

                        int newtotalDataSize =  5 * IMG_SIZE * IMG_SIZE * 3;
                        //FloatBuffer newinputBuffer = ByteBuffer.allocateDirect(totalDataSize * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();



                        int newlabelBufferSize = 5 * 10 * Float.SIZE / Byte.SIZE;
                       // ByteBuffer newlabelBuffer = ByteBuffer.allocateDirect(labelBufferSize).order(ByteOrder.nativeOrder());

                        // Add m_sample data


                        for (Map.Entry<TensorImage, List<Float>> e : newds_n.entrySet()) {
                            TensorImage x_n = e.getKey();
                            List<Float> y_n = e.getValue();
                            new_X_train.add(x_n);
                            new_Y_train.add(y_n);

                        }
                        new_X_train.add(m_data.getImage());
                        new_Y_train.add(m_data.getLabel());
                        float[][][][] newtrainImages = new float[5][160][160][3];
                        float[][] newtrainLabels = new float[5][10];

                        int newnumImages = new_X_train.size();
                        System.out.println(newnumImages);
                        // Convert X_train, Y_train to appropriate types
                        for (int i = 0; i < newnumImages; i++) {
                           // newlabelBuffer.clear();
                           // newinputBuffer.clear();

                            TensorImage imageData = new_X_train.get(i);
                            TensorBuffer tensorBuffer = imageData.getTensorBuffer();

                            float[] imageFloatArray = tensorBuffer.getFloatArray();
                            //newinputBuffer.put(imageFloatArray);


                            //List<Float> newy = new_Y_train.get(i);

                           // for (float label : newy) {
                            //    newlabelBuffer.putFloat(label);
                           // }
                            int index = 0;
                            for (int row = 0; row < 160; row++) {
                                for (int col = 0; col < 160; col++) {
                                    for (int channel = 0; channel < 3; channel++) {
                                        newtrainImages[i][row][col][channel] = imageFloatArray[index++];
                                    }
                                }
                            }
//
                            //List<Float> y = Y_train.get(i);
                            //for (float label : y) {
                            //    labelBuffer.putFloat(label);
                            //}
                            List<Float> label = new_Y_train.get(i);
                            for (int j = 0; j < 10; j++) {
                                newtrainLabels[i][j] = label.get(j);
                            }

                            // Record  loss.

                            //if (i == newnumImages - 1) losses.add(newloss.get(0));
                        }

                       // newinputBuffer.rewind();
                       // newlabelBuffer.rewind();
                        Map<String, Object> newinputs = new HashMap<>();
                        newinputs.put("x", newtrainImages);
                        newinputs.put("y",  newtrainLabels);


                        Map<String, Object> newoutputs = new HashMap<>();
                        FloatBuffer newloss = FloatBuffer.allocate(1);
                        FloatBuffer newprediction = FloatBuffer.allocate(5 * 10 * Float.SIZE / Byte.SIZE);

                        newoutputs.put("loss", newloss);


                        long newstartTime = System.nanoTime();
                        interpreter.runSignature(newinputs, newoutputs,"train");
                        long newendTime = System.nanoTime();
                        long newelapsedTime = newendTime - newstartTime;
                        double newelapsedTimeInSeconds = newelapsedTime / 1_000_000_000.0;
                        time.add(newelapsedTimeInSeconds);

                        losses.add(newloss.get(0));



                        //TODO: EVALUATION ON ds_test_m AND ds_test_n
                        //         accuracy = number_of_correct_predictions / total_samples

                        Map<String, Object> newaccuracies = EvaluateModel(ds_test_n, ds_train_m);
                        for (Map.Entry<String, Object> newentry : newaccuracies.entrySet()) {
                            String key = newentry.getKey();
                            Object value = newentry.getValue();

                            // Add to the appropriate list based on the key
                            if (key.equals("accuracy_m")) {
                                acc_m.add(value);
                            } else if (key.equals("accuracy_n")) {
                                acc_n.add(value);
                            }
                        }

                        System.out.println(newaccuracies);
                        if (abs((float) acc_n.get(acc_n.size() - 1) - (float) acc_m.get(acc_m.size() - 1)) <= 0.02) {
                            System.out.println(
                                    "For Test sets, the accuracies were equal in iteration" + counter + " with total training steps=" + c + ", accuracy=" + acc_m.get(acc_m.size() - 1) + ", and m_samples={}");
                        }
                        c++;
                        System.out.println(
                                "Finished iteration " + counter + ", current loss: " + losses.get(losses.size() - 1));


                    }

                }
                counter++;

            }
        }
        savePrediction(acc_n, "accuracy_n");
        savePrediction(acc_m, "accuracy_m");
        savePrediction(time, "time");
        savePrediction(losses, "loss");

        if (onTrainingCompleteListener != null) {
            onTrainingCompleteListener.onTrainingComplete();
        }
    }



    public int  Classify(TensorImage image){
        int totalDataSize = 1 * 160 * 160 * 3 * Float.SIZE / Byte.SIZE;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(totalDataSize);
        inputBuffer.order(ByteOrder.nativeOrder());


        ByteBuffer imageData = image.getBuffer();
        imageData.rewind();
        inputBuffer.put(imageData);
        inputBuffer.rewind();

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("x", inputBuffer );

//
        Map<String, Object> outputs = new HashMap<>();
        FloatBuffer probabilities = FloatBuffer.allocate(10);
        probabilities.rewind();
        outputs.put("output", probabilities);

        interpreter.runSignature(inputs, outputs,"infer");
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

        return maxIndex ;
    }

    private void savePrediction(List<Object>   accuracies, String numClass) {
        String filename = "document_" + numClass +".txt";
        StringBuilder fileContents = new StringBuilder();
        for (Object prediction : accuracies) {
            fileContents.append(prediction).append("\n");
        }

        try (FileOutputStream fos = context.openFileOutput(filename, Context.MODE_PRIVATE)) {
            fos.write(fileContents.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private  Map<String, Object> EvaluateModel(HashMap<List<Float>, List<TensorImage>> ds_test_n, List<TrainingSample> X_test_m ){
        Map<String, Object> result = new HashMap<>();
        int correct_count_n = 0;
        HashMap<TensorImage,List<Float>> new_ds_test_n = new HashMap<>();
        for(Map.Entry<List<Float>, List<TensorImage>> entry : ds_test_n.entrySet()) {
            List<Float> y = entry.getKey();
            List<TensorImage> x = entry.getValue();
            for (TensorImage x_ : x){
                new_ds_test_n.put(x_,y);
            }
        }
        // shuffle test dataset on n classes

        List<Map.Entry<TensorImage, List<Float>>> entryList = new ArrayList<>(new_ds_test_n.entrySet());
        Collections.shuffle(entryList);
        HashMap<TensorImage, List<Float>> shuffled_ds_test_n = new HashMap<>();
        for (Map.Entry<TensorImage, List<Float>> entry : entryList) {
            shuffled_ds_test_n.put(entry.getKey(), entry.getValue());
        }

        for (Map.Entry<TensorImage, List<Float>> e : shuffled_ds_test_n.entrySet()) {
               TensorImage x =  e.getKey();
               List<Float> y_actual_one_hot = e.getValue() ;
               int y_actual =  Decode(y_actual_one_hot);
               int y_pred =  Classify(x);
               if(y_actual == y_pred){
                   correct_count_n+=1;

               }

        }

        float accuracy_n = (float) correct_count_n /  shuffled_ds_test_n.size();
        result.put("accuracy_n",accuracy_n);

        int correct_count_m = 0;
        for(int i=0; i<X_test_m.size(); i++){
            TrainingSample m_sample =  X_test_m.get(i);
            TensorImage x = m_sample.getImage();
            List<Float> y_one_hot = m_sample.getLabel();
            int y_actual = Decode(y_one_hot);
            int y_pred =  Classify(x);

            if(y_actual == y_pred){
                correct_count_m+= 1;
            }
        }
        float accuracy_m = (float) correct_count_m / X_test_m.size();
        result.put("accuracy_m",accuracy_m);
        return result;
    }

    private int Decode(List<Float> oneHot) {
        for (int i = 0; i < oneHot.size(); i++) {
            if (oneHot.get(i) == 1.0f) {
                return i;
            }
        }
        throw new IllegalArgumentException("Invalid one-hot encoding");
    }
}




