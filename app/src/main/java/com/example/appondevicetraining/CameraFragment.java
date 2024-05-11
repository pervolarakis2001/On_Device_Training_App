package com.example.appondevicetraining;

import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import androidx.activity.result.ActivityResultLauncher;
import androidx.annotation.NonNull;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.view.PreviewView;
import androidx.fragment.app.Fragment;

import android.os.Environment;
import android.text.InputType;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.annotation.SuppressLint;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.content.res.AppCompatResources;
import androidx.camera.core.Preview;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import com.example.appondevicetraining.databinding.FragmentCameraBinding;
import com.google.common.util.concurrent.ListenableFuture;
import androidx.camera.core.ImageCapture;
import android.view.WindowManager;
import android.content.pm.PackageManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import android.provider.MediaStore;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.os.Environment;
import android.content.Intent;
import android.app.Activity;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.model.Model;
import com.example.appondevicetraining.Model_Helper;

public class CameraFragment extends Fragment {

    private List<TrainingImage> m_TrainingImages  = new ArrayList<>(); // store images from gallery
    private List<TrainingSample> TrainingSamples  = new ArrayList<>(); //Store images prepared for training
    private List<List<Float>> y_train_m  = new ArrayList<>(); //Store labels for training
    private List<TensorImage> x_train_m  = new ArrayList<>(); //Store labels for training
    private String className = null;
    private Context context;
    private ImageButton captureBtn;
    private ImageButton addImageBtn;
    private RadioButton TrainingBtn;
    private RadioButton InferenceBtn;
    private Bitmap bitmapBuffer;
    private PreviewView preview;
    private ImageAnalysis imageAnalyzer;
    private Camera camera;
    private ProcessCameraProvider cameraProvider;
    private String previousClass;
    private ExecutorService cameraExecutor;
    private ConcurrentLinkedQueue<String> addSampleRequests;
    private long sampleCollectionButtonPressedTime;
    private boolean isCollectingSamples;
    private Handler sampleCollectionHandler;
    private  Model_Helper modelHelper;
    private static final int REQUEST_CODE_PERMISSION = 100;
    private static final String[] PERMISSIONS_REQUIRED = {android.Manifest.permission.CAMERA};
    private static final String TAG = "Model Personalization";
    private static final int PICK_IMAGES_REQUEST_CODE = 101;


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment

        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        this.context = context;
    }
    @Override
    public void onViewCreated(View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        preview = view.findViewById(R.id.cameraPreview);
        captureBtn = view.findViewById(R.id.capture);
        addImageBtn = view.findViewById(R.id.upload_image);
        TrainingBtn = view.findViewById(R.id.btnTrainingMode);
        InferenceBtn = view.findViewById(R.id.btnInferenceMode);
        modelHelper = new Model_Helper(requireContext());
        startCamera();
        addImageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Check if either TrainingBtn or InferenceBtn is selected
                if (TrainingBtn.isChecked()) {
                    setClassName();
                } else if (InferenceBtn.isChecked()){
                    openGallery();
                } else {
                    Toast.makeText(requireContext(), "Please select a Mode ", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    public void startCamera() {
        int aspectRatio = aspectRatio(preview.getWidth(), preview.getHeight());
        ListenableFuture<ProcessCameraProvider> listenableFuture = ProcessCameraProvider.getInstance(requireContext());
        Log.d(TAG, "camera initialized");
        listenableFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = (ProcessCameraProvider) listenableFuture.get();

                CameraSelector cameraSelector =
                        new CameraSelector.Builder()
                                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                                .build();
                Preview new_preview = new Preview.Builder().setTargetAspectRatio(aspectRatio).build();

                WindowManager windowManager = getActivity().getWindowManager();
                int rotation = windowManager.getDefaultDisplay().getRotation();

                // Create the ImageCapture with the desired configuration
                ImageCapture imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .setTargetRotation(rotation)
                        .build();

                cameraProvider.unbindAll();

                Camera camera = cameraProvider.bindToLifecycle(getViewLifecycleOwner(), cameraSelector, new_preview, imageCapture);

                captureBtn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        takePicture(imageCapture);
                    }
                });
                new_preview.setSurfaceProvider(preview.getSurfaceProvider());

            } catch (Exception e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }


    public void takePicture(ImageCapture imageCapture) {
        final File file = new File(requireContext().getExternalFilesDir(null), System.currentTimeMillis() + ".jpg");
        ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(file).build();
        imageCapture.takePicture(outputFileOptions, Executors.newCachedThreadPool(), new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                Handler mainHandler = new Handler(Looper.getMainLooper());

                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(requireContext(), "Image saved at Gallery ", Toast.LENGTH_SHORT).show();
                    }
                });
                saveImageToGallery(file);
                startCamera();
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Handler mainHandler = new Handler(Looper.getMainLooper());

                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(requireContext(), "Failed to save Image  ", Toast.LENGTH_SHORT).show();
                        Log.d(TAG, "!!!!!!!!!!!!! failed");
                    }
                });
                startCamera();
            }
        });
    }


    private void saveImageToGallery(File file) {
        Uri imageUri;
        ContentResolver contentResolver = requireContext().getContentResolver();

        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.Images.Media.DISPLAY_NAME, System.currentTimeMillis() + ".jpg");
        contentValues.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg"); // corrected MIME type
        contentValues.put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000);

        // Determine the appropriate URI based on the Android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);
            imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
        } else {
            String imagesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString();
            File image = new File(imagesDir, contentValues.getAsString(MediaStore.Images.Media.DISPLAY_NAME));
            imageUri = Uri.fromFile(image);
        }

        try {
            OutputStream outputStream = contentResolver.openOutputStream(Objects.requireNonNull(imageUri));
            FileInputStream fileInputStream = new FileInputStream(file);
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = fileInputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            fileInputStream.close();
            outputStream.close();

        } catch (Exception e) {
            e.printStackTrace();

        }
    }

    private int aspectRatio(int width, int height) {
        double previewRatio = (double) Math.max(width, height) / Math.min(width, height);
        if (Math.abs(previewRatio - 4.0 / 3.0) <= Math.abs(previewRatio - 16.0 / 9.0)) {
            return AspectRatio.RATIO_4_3;
        }
        return AspectRatio.RATIO_16_9;
    }

    private void setClassName(){
        // Show dialog for selecting class
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("Set Class for Training Images");

        // Set up the input
        final EditText input = new EditText(requireContext());
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        // Set up the buttons
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                className = input.getText().toString().trim();

                openGallery();

            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });

        builder.show();

    }
    private void openGallery() {

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        startActivityForResult(intent, PICK_IMAGES_REQUEST_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGES_REQUEST_CODE && resultCode == Activity.RESULT_OK && data != null) {
            if (data.getClipData() != null) {
                // Multiple images selected
                int count = data.getClipData().getItemCount();
                for (int i = 0; i < count; i++) {
                    Uri imageUri = data.getClipData().getItemAt(i).getUri();
                    try {
                        // STORE DATA FOR TRAINING
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), imageUri);

                        m_TrainingImages.add(new TrainingImage(bitmap, className));

                    }catch (Exception e) {
                        // Handle error loading the image
                        e.printStackTrace();
                    }
                }
                if (!m_TrainingImages.isEmpty()) {
                    // Load images of n classes to a dictionary
                    List<Bitmap> ds_n = new ArrayList<>();
                    List<String> Folder_names = new ArrayList<>(Arrays.asList("airplane","automobile","bird","cat","deer","dog","frog","horse","ship"));
                    HashMap<Integer, List<Bitmap>> n_labels_Dictionary = new HashMap<>();
                    int index = 0;
                    for (String folder_name : Folder_names) {
                        List<Bitmap> bitmaps = new ArrayList<>();
                        for (int i = 1; i <= 5; i++) {
                            AssetManager assetManager = context.getAssets();
                            try {
                                    String filename = folder_name + "/image_" + i + ".jpg";
                                    InputStream inputStream = assetManager.open(filename);
                                    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                                    Log.d("Debugging", "is this is null " + bitmap.getHeight()); // Log statement
                                    bitmaps.add(bitmap);
                                } catch (IOException e) {
                                    e.printStackTrace();
                             }
                        }
                        n_labels_Dictionary.put(index++, bitmaps);

                    }

                    // Preprocess Samples
                    List<TrainingSample> ds_train_m =   m_PreprocessSamples(m_TrainingImages);
                    HashMap<List<Float>, List<TensorImage>> ds_train_n = n_PreprocessSamples(n_labels_Dictionary);

                    // do training
                     modelHelper.Training(ds_train_m, ds_train_n );

                }



            } else if (data.getData() != null) {
                // Single image selected
                Uri imageUri = data.getData();
                // STORE DATA FOR INFERENCE
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), imageUri);
                    //preprocessing
                    TensorImage data_infer =  modelHelper.PreprocessImages(bitmap);
                    modelHelper.Classify(data_infer);
                }catch (FileNotFoundException e) {
                    // Handle file not found exception
                    e.printStackTrace();
                } catch (IOException e) {
                    // Handle I/O exception
                    e.printStackTrace();
                }

            }
        }
    }

    // Prepare samples for On Device Training
    private List<TrainingSample> m_PreprocessSamples(List<TrainingImage> TrainingImages ){
        for (TrainingImage trainingImage : TrainingImages) {

            Bitmap bitmap = trainingImage.getBitmap();
            List<Float> label = modelHelper.encoding(9,10);
            TensorImage image = modelHelper.PreprocessImages(bitmap);

            TrainingSamples.add(new TrainingSample(image, label));

        }
        return TrainingSamples;
    }
    private HashMap<List<Float>, List<TensorImage>> n_PreprocessSamples(HashMap<Integer, List<Bitmap>> n_samples ){
        HashMap<List<Float>, List<TensorImage>> ds_n = new HashMap<>();

        Set<Integer> keys = n_samples.keySet();

        for (Integer i : keys) {
            List<Float> label = modelHelper.encoding(i, 10); // Assuming encoding returns a List<Float>
            List<Bitmap> value = n_samples.get(i);
            List<TensorImage>  n_preprocessed = new ArrayList<>();

            for(Bitmap bitmap : value){
                TensorImage image = modelHelper.PreprocessImages(bitmap);
                n_preprocessed.add(image);
            }
            ds_n.put(label, n_preprocessed);
        }
        return  ds_n;
    }



}